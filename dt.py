import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
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
    plt.show()



if __name__ == "__main__":
    main()