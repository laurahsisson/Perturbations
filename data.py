import pandas as pd

import smiles
import torch


# x data is formatted as celltype (string) and smiles
# y data is tensor of length 18211 containing DGE for all genes
def get_train():
    print("Loading train data.")
    df = pd.read_parquet("data/de_train.parquet")
    x = []
    y = []

    for index, row in df.iterrows():
        ct = row["cell_type"]
        sm = smiles.get(row["sm_name"])
        x.append((ct, sm))
        y.append(row[5:].values.astype(float))

    return x, y


# returns x as above
def get_test():
    print("Loading test data.")
    df = pd.read_csv("data/id_map.csv")
    x = []
    for index, row in df.iterrows():
        ct = row["cell_type"]
        sm = smiles.get(row["sm_name"])
        x.append((ct, sm))

    return x


if __name__ == "__main__":
    train_x, train_y = get_train()
    print(train_x[0])
    print(torch.from_numpy(train_y[0]))

    test_x = get_test()
    print(test_x[0])
