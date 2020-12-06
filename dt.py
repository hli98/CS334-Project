import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


def main():
    #Read Data
    #train_x = pd.read_csv("train_x.csv")
    #fn = train_x.columns.tolist()
    #train_x = train_x.to_numpy()
    train_x = pd.read_csv("train_x.csv").to_numpy()
    train_y = pd.read_csv("train_y.csv").to_numpy()
    test_x = pd.read_csv("test_x.csv").to_numpy()
    test_y = pd.read_csv("test_y.csv").to_numpy()

    '''
    #look for row indices where there is no prior order
    row_indices = []
    row_index = 0
    #Actually, every order in the training dataset has a prior order
    for row in x:
        if row[4] == -1:
            row_indices.append(row_index)
        row_index += 1
    '''
    '''
    #Train-Test split   
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.33, random_state=42)
    '''


    #Decision Tree
    clf = tree.DecisionTreeClassifier()#(max_depth=26)#, min_samples_leaf=10000)
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



    #for i in range(len(dtyhat)):
        #if i in row_indices:
            #dtyhat[i] = 0
    #pd.DataFrame(train_yhat).to_csv("train_yhat.csv", index = False)
    #print("train y length",len(train_yhat))

    '''
    #Plot DT
    fig, axes = plt.subplots(nrows = 1,ncols = 1, dpi=100, figsize=(12, 12))
    tree.plot_tree(dtclf, feature_names = fn, ax=axes)
    plt.savefig("dt.png", fontsize=12)
    plt.show()
    '''


if __name__ == "__main__":
    main()