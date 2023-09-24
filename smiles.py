import pandas as pd
import json
import torch

fname = 'smiles.json'

mapping = dict()
all_smiles = list()


def build():
    global mapping

    df = pd.read_parquet("data/de_train.parquet")
    for index, row in df.iterrows():
        mapping[row["sm_name"]] = row["SMILES"]

    with open(fname, 'w') as f:
        json.dump(mapping, f)


def load():
    global mapping, all_smiles

    print("Loading smiles from disk.")
    with open(fname, 'r') as f:
        mapping = json.load(f)

    all_smiles = list(sorted(mapping.values()))


def get(sm_name):
    if not mapping:
        load()

    return mapping[sm_name]


def one_hot(sm_name):
    if not mapping:
        load()

    idx = torch.tensor(all_smiles.index(sm_name))
    return torch.nn.functional.one_hot(idx, num_classes=len(all_smiles))


if __name__ == "__main__":
    build()
