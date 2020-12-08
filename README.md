# cs334-project
Please follow the order of following step to execute the codes.

1.  Download every file in this repository
2.  Create a directory named "src" under the directory where you put these files.
3.  Go to https://www.kaggle.com/c/instacart-market-basket-analysis/data
4.  Download all the data files on this webpage, unzip and put these .csv files under src.
5.  Run "InstacartAnalysisExploration.ipynb", it explores the original data.
6.  Run "preprocessing.ipynb", it preprocess the data under "src" directory and output new data files required for "dt.py" and "rf.py".
7.  Run "dt.py" and "rf.py" for machine learning. You may run these two programs in any order after running "preprocessing.ipynb" first. "dt.py" uses the decision tree algorithm while "rf.py" uses random forest.
