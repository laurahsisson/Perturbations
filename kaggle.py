import torch

import pandas as pd
import tqdm
import numpy as np

LINCS_TSM_IDX = 6
lincs_df = pd.read_parquet("data/lincs.parquet")
lincs_x = lincs_df.iloc[:,:LINCS_TSM_IDX]
lincs_genes = lincs_df.columns[LINCS_TSM_IDX:]

TRAIN_TSM_IDX = 5
kaggle_df = pd.read_parquet("data/de_train.parquet")
shared_genes = kaggle_df.columns[kaggle_df.columns.isin(lincs_genes)]

sm_df = kaggle_df[["sm_name","SMILES"]].set_index("sm_name")
sm_dict = sm_df.to_dict()["SMILES"]

test_df = pd.read_csv("data/id_map.csv")

def find_control(df,cell_col_name,cell_value):
    # Could either return random or mean.
    valid_columns = df[(df[cell_col_name]==cell_value) & (df["control"])]
    return valid_columns.sample(1).iloc[0]

def get_transcriptome(row):
    return row[shared_genes].to_numpy().astype(float)

def make_datapoint(tsm_idx,df,cell_col_name,row):
    data = row[:tsm_idx].to_dict()
    data["pre_treatment"] = get_transcriptome(find_control(df,cell_col_name,row[cell_col_name]))
    data["post_treatment"] = get_transcriptome(row)
    return data

def test_datapoint(cell_col_name,row):
    data = row.to_dict()
    data["SMILES"] = sm_dict[row["sm_name"]]
    data["pre_treatment"] = get_transcriptome(find_control(kaggle_df,cell_col_name,row[cell_col_name]))
    return data

# print(get_transcriptome(find_control(lincs_df,"cell_id","A375")))
# print(get_transcriptome(find_control(kaggle_df,"cell_type","B cells")))

def make_data():
    lincs_cmpd_df = lincs_df[~lincs_df["control"]]
    lincs_dps = []
    for _, row in tqdm.tqdm(lincs_cmpd_df.iterrows(),total=lincs_cmpd_df.shape[0]):
        lincs_dps.append(make_datapoint(LINCS_TSM_IDX,lincs_df,"cell_id",row))
    torch.save(lincs_dps,"data/lincs_pretreatment.pt")

    kaggle_cmpd_df = kaggle_df[~kaggle_df["control"]]
    kaggle_dps = []
    for _, row in tqdm.tqdm(kaggle_cmpd_df.iterrows(),total=kaggle_cmpd_df.shape[0]):
        data = make_datapoint(TRAIN_TSM_IDX,kaggle_df,"cell_type",row)
        # Also include the full transcriptome so that this can be used for imputing.
        data["transcriptome"] = row[TRAIN_TSM_IDX:].to_numpy().astype(float)
        kaggle_dps.append(data)
    torch.save(kaggle_dps,"data/kaggle_pretreatment.pt")

    test_dps = []
    for _, row in tqdm.tqdm(test_df.iterrows(),total=test_df.shape[0]):
        test_dps.append(test_datapoint("cell_type",row))
    torch.save(test_dps,"data/test_pretreatment.pt")

if __name__ == "__main__":
    make_data()