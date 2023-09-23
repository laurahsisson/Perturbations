import rdkit
import rdkit.Chem.rdFingerprintGenerator
import pandas as pd
import data
import celltype
import numpy as np
import tqdm

import sklearn
import sklearn.linear_model
import submission

mfpgen = rdkit.Chem.rdFingerprintGenerator.GetMorganGenerator(radius=4,fpSize=2048)

# https://github.com/rdkit/rdkit/discussions/3863
def to_numpy(fp):
    return np.frombuffer(bytes(fp.ToBitString(), 'utf-8'), 'u1') - ord('0')

def get_fingerprint(smiles):
    mol = rdkit.Chem.MolFromSmiles(smiles)
    fp = mfpgen.GetFingerprint(mol)
    return to_numpy(fp)

def convert_to_input(x_entry):
    cell_type, smiles = x_entry
    ct = celltype.one_hot(cell_type)
    mfp = get_fingerprint(smiles)
    return np.concatenate([ct,mfp])

train_x_entries, train_y = data.get_train()
test_x_entries = data.get_test()

train_x = np.stack([convert_to_input(x_entry) for x_entry in train_x_entries])
train_y = np.stack(train_y)

test_x = np.stack([convert_to_input(x_entry) for x_entry in test_x_entries])

model = sklearn.linear_model.LinearRegression().fit(train_x,train_y)
print(train_y[0].tolist())
print(model.score(train_x,train_y))
test_pred = model.predict(test_x)
print(f"Made predictions of shape {test_pred.shape}.")
submission.make("fingerprint",test_pred)

