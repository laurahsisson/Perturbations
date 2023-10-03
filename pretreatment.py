import torch

import pandas as pd
import tqdm

TSM_IDX = 6
df = pd.read_parquet("data/lincs.parquet")

xdf,ydf = df.iloc[:,:TSM_IDX], df.iloc[:,TSM_IDX:]

def find_control(cell_id):
    # Could either return random or mean.
    valid_columns = df[(df["cell_id"]==cell_id) & (df["control"])]
    return valid_columns.sample(1).iloc[0]

def get_transcriptome(row):
    return torch.tensor(row[TSM_IDX:].to_numpy().astype(float))

def make_datapoint(row):
    data = row[:TSM_IDX].to_dict()
    data["pre_treatment"] = get_transcriptome(find_control(row["cell_id"]))
    data["post_treatment"] = get_transcriptome(row)
    return data


cmpd_df = df[~df["control"]]
dps = []
for _, row in tqdm.tqdm(cmpd_df.iterrows(),total=cmpd_df.shape[0]):
    dps.append(make_datapoint(row))
torch.save(dps,"data/pretreatment.pt")