import pandas as pd
from sklearn import ensemble
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def main():
    #read data
    train_x = pd.read_csv("train_x.csv")
    fn = train_x.columns.tolist()
    train_x = train_x.to_numpy()
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


    
    #Showing the first 5 rfs
    for index in range(0, 5):
        rules = tree.export_text(decision_tree = clf.estimators_[index], feature_names = fn)
        filename = "rfrules" + str(index + 1) + ".txt"
        with open(filename, "w") as text_file:
            text_file.write(rules)


if __name__ == "__main__":
    main()