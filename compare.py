import pandas as pd 
import numpy as np

def main():
    #read
    y = pd.read_csv("train_y.csv").to_numpy()
    dt_y = pd.read_csv("dt_yhat.csv").to_numpy()
    rf_y = pd.read_csv("rf_yhat.csv").to_numpy()

    diff_1 = 0 # diff between y and dt prediction
    diff_2 = 0 # diff between y and rf prediction
    diff_3 = 0 # diff between dt and rf prediction

    num_rows = len(y)

    for i in range(num_rows):
        if y[i] != dt_y[i]: diff_1 += 1
        if y[i] != rf_y[i]: diff_2 += 1
        if dt_y[i] != rf_y[i]: diff_3 += 1
    print("diff_1:", diff_1 / num_rows)
    print("diff_2:", diff_2 / num_rows)
    print("diff_3:", diff_3 / num_rows)

if __name__ == "__main__":
    main()