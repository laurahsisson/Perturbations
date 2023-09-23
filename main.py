import pandas as pd

import smiles
import celltype

# sm_names_to_smiles = dict()
df = pd.read_parquet("data/de_train.parquet")
# for index, row in df.iterrows():
#     # sm_names_to_smiles[row["sm_name"]] = row["SMILES"]



# test = pd.read_csv("data/id_map.csv")
# for index, row in test.iterrows():
#     print(celltype.one_hot(row["cell_type"]))

# example = pd.read_csv("data/sample_submission.csv")


# print(test.info)
# print()
# print()
# print(example.info)