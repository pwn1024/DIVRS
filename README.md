# DIVRS

First download and unzip the dataset, which is preprocessed in a way described in detail in the paper：
Movielens-10M and Douban-Movie

Second, please start the Visdom server:
Run the following command in your command line.
visdom -port 33336

Third, modify the dataset path and working path in the following files according to your actual setup:
app.py, const.py

Finally, execute the following command to run the application:
python app.py --flagfile ./config/xxx.cfg

The code is based on [DICE](https://github.com/tsinghua-fib-lab/DICE)
