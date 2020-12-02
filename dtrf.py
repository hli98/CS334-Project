import pandas as pd
import matplotlib as plt
from sklearn import tree
from sklearn import ensemble
from sklearn.model_selection import train_test_split


def main():
    #read data
    x = pd.read_csv("train_x.csv").to_numpy()
    y = pd.read_csv("train_y.csv").to_numpy()
    #print("y length",len(y))
    #hide next line, undo it when generate formal output
    #train_info = pd.read_csv("train_info.csv").to_numpy()

    #look for row indices where there is no prior order
    row_indices = []
    row_index = 0
    #Actually, every order in the training dataset has a prior order
    for row in x:
        if row[4] == -1:
            row_indices.append(row_index)
        row_index += 1


    #Train-Test split   
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.33, random_state=42)



    #Decision Tree
    dtclf = tree.DecisionTreeClassifier()
    dtclf.fit(train_x, train_y.ravel())
    dtyhat = dtclf.predict(train_x)
    for i in range(len(dtyhat)):
        if i in row_indices:
            dtyhat[i] = 0
    pd.DataFrame(dtyhat).to_csv("dt_yhat.csv", index = False)
    print("dt y length",len(dtyhat))

    #Plot DT
    tree.plot_tree(dtclf)

    #Random Forest
    rfclf = ensemble.RandomForestClassifier()
    rfclf.fit(train_x, train_y.ravel())
    rfyhat = rfclf.predict(train_x)
    for i in range(len(rfyhat)):
        if i in row_indices:
            rfyhat[i] = 0
    pd.DataFrame(rfyhat).to_csv("rf_yhat.csv", index = False)
    print("rf y length",len(rfyhat))

    #Plot RF
    #Showing the first 5 rfs
    fn = train_x.columns.values.tolist()
    fig, axes = plt.subplots(nrows = 1,ncols = 5,figsize = (10,2), dpi=900)
    for index in range(0, 5):
        tree.plot_tree(rfclf.estimators_[index],
                    feature_names = fn, 
                    filled = True,
                    ax = axes[index])
        axes[index].set_title('Estimator: ' + str(index), fontsize = 11)




if __name__ == "__main__":
    main()