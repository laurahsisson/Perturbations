import torch

import pandas as pd
import tqdm

LINCS_TSM_IDX = 6
lincs_df = pd.read_parquet("data/lincs.parquet")
lincs_x = lincs_df.iloc[:,:LINCS_TSM_IDX]
lincs_genes = lincs_df.columns[LINCS_TSM_IDX:]

TRAIN_TSM_IDX = 5
train_df = pd.read_parquet("data/de_train.parquet")
train_x = train_df.iloc[:,:TRAIN_TSM_IDX]

shared_genes = train_df.columns[train_df.columns.isin(lincs_genes)]

print(shared_genes)
lincs_y = lincs_df[shared_genes]
train_y = train_df[shared_genes]

print("lincs_x",lincs_x.shape,lincs_x.columns)
print("train_x",train_x.shape,train_x.columns)
print("lincs_y",lincs_y.shape)
print("train_y",train_y.shape)

def find_control(df,col_name,col_value):
    # Could either return random or mean.
    valid_columns = df[(df[col_name]==col_value) & (df["control"])]
    return valid_columns.sample(1).iloc[0]

print(find_control(lincs_df,"cell_id","A375"))
print(find_control(train_df,"cell_type","B cells"))

