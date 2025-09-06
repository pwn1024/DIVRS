# coding=utf-8
# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=import-error

import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import torch.nn.functional as F

import dgl.function as fn

import utils

from deprecated import deprecated
from tqdm import tqdm

import random


class DIVRS(nn.Module):

    def __init__(self, num_users, num_items, embedding_size, dis_loss, dis_pen, int_weight, pop_weight):

        super(DIVRS, self).__init__()

        self.uiv = Parameter(torch.FloatTensor(num_users, embedding_size))
        self.iiv = Parameter(torch.FloatTensor(num_items, embedding_size))

        self.users_iv = Parameter(torch.FloatTensor(num_users, embedding_size))
        self.users_ori = Parameter(torch.FloatTensor(num_users, embedding_size))
        self.items_iv = Parameter(torch.FloatTensor(num_items, embedding_size))
        self.items_ori = Parameter(torch.FloatTensor(num_items, embedding_size))

        params = torch.ones(3, requires_grad=True)
        self.params = torch.nn.Parameter(params)

        self.iv_weight = int_weight
        # self.cos_weight = pop_weight

        self.mlp = torch.nn.Linear(128, 128)
        self.relu = torch.nn.ReLU()

        if dis_loss == 'L1':
            self.criterion_discrepancy = nn.L1Loss()
        elif dis_loss == 'L2':
            self.criterion_discrepancy = nn.MSELoss()
        elif dis_loss == 'dcor':
            self.criterion_discrepancy = self.dcor

        self.dis_pen = dis_pen

        self.init_params()

    def adapt(self, epoch, decay):
        # print(self.iv_weight)
        self.iv_weight = self.iv_weight * decay
        # self.cos_weight = self.cos_weight * decay


    def dcor(self, x, y):

        a = torch.norm(x[:, None] - x, p=2, dim=2)
        b = torch.norm(y[:, None] - y, p=2, dim=2)

        A = a - a.mean(dim=0)[None, :] - a.mean(dim=1)[:, None] + a.mean()
        B = b - b.mean(dim=0)[None, :] - b.mean(dim=1)[:, None] + b.mean()

        n = x.size(0)

        dcov2_xy = (A * B).sum() / float(n * n)
        dcov2_xx = (A * A).sum() / float(n * n)
        dcov2_yy = (B * B).sum() / float(n * n)
        dcor = -torch.sqrt(dcov2_xy) / torch.sqrt(torch.sqrt(dcov2_xx) * torch.sqrt(dcov2_yy))

        return dcor

    def init_params(self):

        stdv = 1. / math.sqrt(self.users_iv.size(1))
        self.users_iv.data.uniform_(-stdv, stdv)
        self.users_ori.data.uniform_(-stdv, stdv)
        self.items_iv.data.uniform_(-stdv, stdv)
        self.items_ori.data.uniform_(-stdv, stdv)
        self.uiv.data.uniform_(-stdv, stdv)
        self.iiv.data.uniform_(-stdv, stdv)

    def bpr_loss(self, p_score, n_score):

        return -torch.mean(torch.log(torch.sigmoid(p_score - n_score)))

    def mask_bpr_loss(self, p_score, n_score, mask):

        return -torch.mean(mask * torch.log(torch.sigmoid(p_score - n_score)))

    def forward(self, user, item_p, item_n, mask):

        uiv = self.uiv[user]
        iiv = self.iiv[item_p]

        users_iv = self.users_iv[user]
        users_ori = self.users_ori[user]
        items_p_iv = self.items_iv[item_p]
        items_p_ori = self.items_ori[item_p]
        items_n_iv = self.items_iv[item_n]
        items_n_ori = self.items_ori[item_n]

        p_score_ori = torch.sum(users_ori * items_p_ori, 2)
        n_score_ori = torch.sum(users_ori * items_n_ori, 2)

        loss_ori = self.bpr_loss(p_score_ori, n_score_ori)


        loss_cos_u = -torch.mean(torch.abs(torch.cosine_similarity(uiv[:, -1], users_ori[:, -1], dim=0)))
        loss_cos_i = -torch.mean(torch.abs(torch.cosine_similarity(iiv[:, -1], items_p_ori[:, -1], dim=0))

        device = uiv.device
        uiv = uiv.cpu()
        iiv = iiv.cpu()
        try:
            pinverse_uiv = torch.pinverse(torch.where(torch.isnan(uiv[:, -1]), torch.full_like(uiv[:, -1], 0), uiv[:, -1]))
            pinverse_iiv = torch.pinverse(torch.where(torch.isnan(iiv[:, -1]), torch.full_like(iiv[:, -1], 0), iiv[:, -1]))
        except:
            print(uiv[:, -1])
            print(iiv[:, -1])
            print(torch.isnan(uiv[:, -1]).any())
            print(torch.isinf(uiv[:, -1]).any())
            print("===========================")
            print(torch.isnan(iiv[:, -1]).any())
            print(torch.isinf(iiv[:, -1]).any())
        uiv = uiv.to(device)
        iiv = iiv.to(device)
        pinverse_iiv = pinverse_iiv.to(device)
        pinverse_uiv = pinverse_uiv.to(device)


        users_iv_causal = torch.matmul(uiv[:, -1], torch.matmul(pinverse_uiv, users_iv[:, -1]))
        items_p_iv_causal = torch.matmul(iiv[:, -1], torch.matmul(pinverse_iiv, items_p_iv[:, -1]))


        users_iv_res = users_iv[:, -1] - users_iv_causal
        items_p_iv_res = items_p_iv[:, -1] - items_p_iv_causal



        users_iv = 0.8 * users_iv_causal + 0.2 * users_iv_res
        items_p_iv = 0.8 * items_p_iv_causal + 0.2 * items_p_iv_res


        users_iv = users_iv.unsqueeze(1).repeat(1, 4, 1)
        items_p_iv = items_p_iv.unsqueeze(1).repeat(1, 4, 1)


        p_score_iv = torch.sum(users_iv * items_p_iv, 2)
        n_score_iv = torch.sum(users_iv * items_n_iv, 2)
        loss_iv = self.bpr_loss(p_score_iv, n_score_iv)

        loss = loss_iv + loss_ori + (loss_cos_i + loss_cos_u)


        return loss

    def get_item_embeddings(self):

        item_embeddings = self.items_iv
        return item_embeddings.detach().cpu().numpy().astype('float32')

    def get_user_embeddings(self):

        user_embeddings = self.users_iv
        return user_embeddings.detach().cpu().numpy().astype('float32')


class DIVRS_GCN(nn.Module):

    def __init__(self, num_users, num_items, embedding_size, num_layers, dropout, dis_loss, dis_pen, int_weight,
                 pop_weight):

        super(DIVRS_GCN, self).__init__()

        self.n_user = num_users
        self.n_item = num_items

        self.int_weight = int_weight
        self.pop_weight = pop_weight

        self.embeddings_iv = Parameter(torch.FloatTensor(num_users + num_items, embedding_size))
        self.embeddings_ori = Parameter(torch.FloatTensor(num_users + num_items, embedding_size))
        self.iv = Parameter(torch.FloatTensor(num_users + num_items, embedding_size))

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(LGConv(embedding_size, embedding_size, 1))

        self.dropout = dropout

        if dis_loss == 'L1':
            self.criterion_discrepancy = nn.L1Loss()
        elif dis_loss == 'L2':
            self.criterion_discrepancy = nn.MSELoss()
        elif dis_loss == 'dcor':
            self.criterion_discrepancy = self.dcor

        self.dis_pen = dis_pen

        self.init_params()

    def dcor(self, x, y):

        a = torch.norm(x[:, None] - x, p=2, dim=2)
        b = torch.norm(y[:, None] - y, p=2, dim=2)

        A = a - a.mean(dim=0)[None, :] - a.mean(dim=1)[:, None] + a.mean()
        B = b - b.mean(dim=0)[None, :] - b.mean(dim=1)[:, None] + b.mean()

        n = x.size(0)

        dcov2_xy = (A * B).sum() / float(n * n)
        dcov2_xx = (A * A).sum() / float(n * n)
        dcov2_yy = (B * B).sum() / float(n * n)
        dcor = -torch.sqrt(dcov2_xy) / torch.sqrt(torch.sqrt(dcov2_xx) * torch.sqrt(dcov2_yy))

        return dcor

    def init_params(self):

        stdv = 1. / math.sqrt(self.embeddings_iv.size(1))
        self.embeddings_iv.data.uniform_(-stdv, stdv)
        self.embeddings_ori.data.uniform_(-stdv, stdv)
        self.iv.data.uniform_(-stdv, stdv)

    def adapt(self, epoch, decay):

        self.int_weight = self.int_weight * decay
        self.pop_weight = self.pop_weight * decay

    def bpr_loss(self, p_score, n_score):

        return -torch.mean(torch.log(torch.sigmoid(p_score - n_score)))

    def mask_bpr_loss(self, p_score, n_score, mask):

        return -torch.mean(mask * torch.log(torch.sigmoid(p_score - n_score)))

    def forward(self, user, item_p, item_n, mask, graph, training=True):

        features_iv = [self.embeddings_iv]
        h = self.embeddings_iv
        for layer in self.layers:
            h = layer(graph, h)
            h = F.dropout(h, p=self.dropout, training=training)
            features_iv.append(h)

        features_iv = torch.stack(features_iv, dim=2)
        features_iv = torch.mean(features_iv, dim=2)

        features_ori = [self.embeddings_ori]
        h = self.embeddings_ori
        for layer in self.layers:
            h = layer(graph, h)
            h = F.dropout(h, p=self.dropout, training=training)
            features_ori.append(h)

        features_ori = torch.stack(features_ori, dim=2)
        features_ori = torch.mean(features_ori, dim=2)

        iv = [self.iv]
        h = self.iv
        for layer in self.layers:
            h = layer(graph, h)
            h = F.dropout(h, p=self.dropout, training=training)
            iv.append(h)

        iv = torch.stack(iv, dim=2)
        iv = torch.mean(iv, dim=2)


        item_p = item_p + self.n_user
        item_n = item_n + self.n_user

        users_iv = features_iv[user]
        users_ori = features_ori[user]
        uiv = iv[user]

        items_p_iv = features_iv[item_p]
        items_p_ori = features_ori[item_p]
        iiv = iv[item_p]

        items_n_iv = features_iv[item_n]
        items_n_ori = self.embeddings_ori[item_n]

        p_score_ori = torch.sum(users_ori * items_p_ori, 2)
        n_score_ori = torch.sum(users_ori * items_n_ori, 2)

        loss_ori = self.bpr_loss(p_score_ori, n_score_ori)

        loss_cos_u = -torch.mean(torch.abs(torch.cosine_similarity(uiv[:, -1], users_ori[:, -1], dim=0)))
        loss_cos_i = -torch.mean(torch.abs(torch.cosine_similarity(iiv[:, -1], items_p_ori[:, -1], dim=0)))

        device = self.iv.device
        uiv = uiv.cpu()
        iiv = iiv.cpu()
        try:
            pinverse_uiv = torch.pinverse(
                torch.where(torch.isnan(uiv[:, -1]), torch.full_like(uiv[:, -1], 0), uiv[:, -1]))
            pinverse_iiv = torch.pinverse(
                torch.where(torch.isnan(iiv[:, -1]), torch.full_like(iiv[:, -1], 0), iiv[:, -1]))
        except:
            print(uiv[:, -1])
            print(iiv[:, -1])
            print(torch.isnan(uiv[:, -1]).any())
            print(torch.isinf(uiv[:, -1]).any())
            print("===========================")
            print(torch.isnan(iiv[:, -1]).any())
            print(torch.isinf(iiv[:, -1]).any())
        uiv = uiv.to(device)
        iiv = iiv.to(device)
        pinverse_iiv = pinverse_iiv.to(device)
        pinverse_uiv = pinverse_uiv.to(device)


        users_iv_causal = torch.matmul(uiv[:, -1], torch.matmul(pinverse_uiv, users_iv[:, -1]))
        items_p_iv_causal = torch.matmul(iiv[:, -1], torch.matmul(pinverse_iiv, items_p_iv[:, -1]))


        users_iv_res = users_iv[:, -1] - users_iv_causal
        items_p_iv_res = items_p_iv[:, -1] - items_p_iv_causal

        users_iv = 0.8 * users_iv_causal + 0.2 * users_iv_res
        items_p_iv = 0.8 * items_p_iv_causal + 0.2 * items_p_iv_res


        #OPR
        u_dot_product = torch.sum(uiv[:, -1] * users_iv_res, dim=1)
        u_reg_loss = torch.mean(u_dot_product ** 2)
        i_dot_product = torch.sum(iiv[:, -1] * items_p_iv_res, dim=1)
        i_reg_loss = torch.mean(i_dot_product ** 2)
        reg_loss = 0.5 * (u_reg_loss + i_reg_loss)


        users_iv = users_iv.unsqueeze(1).repeat(1, 4, 1)
        items_p_iv = items_p_iv.unsqueeze(1).repeat(1, 4, 1)


        p_score_iv = torch.sum(users_iv * items_p_iv, 2)
        n_score_iv = torch.sum(users_iv * items_n_iv, 2)
        loss_iv = self.bpr_loss(p_score_iv, n_score_iv)


        loss = loss_iv + loss_ori + (loss_cos_i + loss_cos_u) + reg_loss

        return loss

    def get_embeddings(self, graph):

        features_iv = [self.embeddings_iv]
        h = self.embeddings_iv
        for layer in self.layers:
            h = layer(graph, h)
            features_iv.append(h)

        features_iv = torch.stack(features_iv, dim=2)
        features_iv = torch.mean(features_iv, dim=2)

        users_iv = features_iv[:self.n_user]
        items_iv = features_iv[self.n_user:]

        return items_iv.detach().cpu().numpy().astype('float32'), users_iv.detach().cpu().numpy().astype('float32')
