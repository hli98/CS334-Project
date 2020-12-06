import pandas as pd
import matplotlib.pyplot as plt
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def main():
    #read data
    train_x = pd.read_csv("train_x.csv").to_numpy()
    train_y = pd.read_csv("train_y.csv").to_numpy()
    test_x = pd.read_csv("test_x.csv").to_numpy()
    test_y = pd.read_csv("test_y.csv").to_numpy()

    #Random Forest
    clf = ensemble.RandomForestClassifier()
    clf.fit(train_x, train_y.ravel())
    train_yhat = clf.predict(train_x)
    test_yhat = clf.predict(test_x)

    #Metrics
    train_acc = accuracy_score(train_y, train_yhat)
    test_acc = accuracy_score(test_y, test_yhat)
    train_f1 = f1_score(train_y, train_yhat)
    test_f1 = f1_score(test_y, test_yhat)
    

    #Printing out Metrics
    print("train_acc", train_acc)
    print("test_acc", test_acc)
    print("train_f1", train_f1)
    print("test_f1", test_f1)

    '''
    #Showing the first 5 rfs
    fn = train_x.columns.values.tolist()
    fig, axes = plt.subplots(nrows = 1,ncols = 5,figsize = (10,2), dpi=900)
    for index in range(0, 5):
        tree.plot_tree(rfclf.estimators_[index],
                    feature_names = fn, 
                    filled = True,
                    ax = axes[index])
        axes[index].set_title('Estimator: ' + str(index), fontsize = 11)
    plt.show()
    '''



if __name__ == "__main__":
    main()