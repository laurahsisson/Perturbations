import pandas as pd

import tqdm

import numpy as np



def make(fname,pred):
    df = pd.read_csv("data/sample_submission.csv")
    # Expecting the same number of predictions
    assert pred.shape[0] == df.shape[0]
    # Predictions will have 1 less element (because they do not contain id)
    assert pred.shape[1] == df.shape[1] - 1
    for i, entry in enumerate(tqdm.tqdm(pred)):
        rowval = entry
        df.iloc[i,1:] = rowval

    with open(f"submissions/{fname}.csv","w") as f:
        df.to_csv(f,index=False)
