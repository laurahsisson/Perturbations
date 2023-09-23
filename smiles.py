import pandas as pd
import json

fname = 'smiles.json'

mapping = dict()

def build():
    global mapping

    df = pd.read_parquet("data/de_train.parquet")
    for index, row in df.iterrows():
        mapping[row["sm_name"]] = row["SMILES"]

    with open(fname, 'w') as f:
        json.dump(mapping, f)


def load():
    global mapping

    print("Loading smiles from disk.")
    with open(fname,'r') as f:
        mapping = json.load(f)


def get(sm_name):
    if not mapping:
        load()

    return mapping[sm_name]

if __name__ == "__main__":
    build()