import rdkit
import rdkit.Chem.rdFingerprintGenerator
import pandas as pd
import data
import celltype
import tqdm
import numpy as np
import sklearn
import sklearn.linear_model
import sklearn.kernel_ridge
import sklearn.decomposition
import submission
import smiles
import mrrmse
import torch

import matplotlib.pyplot as plt
import sklearn.metrics

# https://github.com/rdkit/rdkit/discussions/3863


def to_numpy(fp):
    return np.frombuffer(bytes(fp.ToBitString(), 'utf-8'), 'u1') - ord('0')


def get_fingerprint(smls):
    mol = rdkit.Chem.MolFromSmiles(smls)
    fp = mfpgen.GetFingerprint(mol)
    return to_numpy(fp)


def to_tensor(lst):
    return torch.tensor(np.array(lst),dtype=torch.float32)


def convert_to_input(fpfn, x_entry):
    ct, smls = x_entry
    ct = celltype.one_hot(ct)
    mfp = fpfn(smls)
    return np.concatenate([ct, mfp])

def weight(x_entry):
    ct, smiles = x_entry
    if celltype.is_evaluation(ct):
        return 1000
    return 1

fp_size = 2048
embed_size = 256
mfpgen = rdkit.Chem.rdFingerprintGenerator.GetMorganGenerator(radius=4,
                                                              fpSize=2048)

train_x_entries, train_y = data.get_train(data.Include.All)
train_weights = torch.tensor([weight(x_entry) for x_entry in train_x_entries])
train_x = to_tensor([convert_to_input(get_fingerprint,entry) for entry in train_x_entries])
train_y = to_tensor(train_y)

eval_x_entries, eval_y = data.get_train(data.Include.Evaluation)
eval_x = to_tensor([convert_to_input(get_fingerprint,entry) for entry in eval_x_entries])
eval_y = to_tensor(eval_y)

model = torch.nn.Sequential(torch.nn.Linear(train_x.shape[1], embed_size), torch.nn.ReLU(
), torch.nn.Dropout(p=0), torch.nn.Linear(embed_size, train_y.shape[1]))

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=.001)


def weighted_mse(y_pred, y_true, weights):
    squared_errors = torch.square(y_pred - y_true)
    weighted_squared_errors = squared_errors * weights.unsqueeze(1)
    loss = torch.mean(weighted_squared_errors)
    return loss

best_loss = float('inf')
model.train()
for _ in tqdm.tqdm(range(10000)):
    pred = model(train_x)
    loss = weighted_mse(pred,train_y,train_weights)
    if loss < best_loss:
        best_loss = loss
    else:
        break
    # print(loss)
    loss.backward()
    optimizer.step()

# tensor(1.1487) with weighted one_hot and .0001
# tensor(1.2076) with one_hot and .0001
# tensor(1.2931) with fingerprint and .00001
with torch.no_grad():
    model.eval()
    print(mrrmse.vectorized(model(eval_x),eval_y))