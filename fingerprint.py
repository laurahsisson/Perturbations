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
import submission

import matplotlib.pyplot as plt
import sklearn.metrics

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
    return np.concatenate([ct, mfp])

def weight(x_entry):
    cell_type, smiles = x_entry
    if celltype.is_evaluation(cell_type):
        return 1000
    return 1

def score(y,pred):
    return sklearn.metrics.mean_squared_error(y,pred,squared=False)

train_x_entries, train_y = data.get_train(data.Include.All)
eval_x_entries, eval_y = data.get_train(data.Include.Evaluation)
        
mfpgen = rdkit.Chem.rdFingerprintGenerator.GetMorganGenerator(radius=4,
                                                      fpSize=2048)
train_x = np.stack([convert_to_input(x_entry) for x_entry in train_x_entries])
train_weights = np.stack([weight(x_entry) for x_entry in train_x_entries])
train_y = np.stack(train_y)

eval_x = np.stack([convert_to_input(x_entry) for x_entry in eval_x_entries])
eval_y = np.stack(eval_y)

# print("Training model.")
model = sklearn.kernel_ridge.KernelRidge(kernel='linear').fit(train_x, train_y,sample_weight=train_weights)

train_pred = model.predict(train_x)
eval_pred = model.predict(eval_x)
print("Eval",score(eval_y,eval_pred))

test_x_entries = data.get_test()
test_x = np.stack([convert_to_input(x_entry) for x_entry in test_x_entries])
test_pred = model.predict(test_x)
print(f"Made predictions of shape {test_pred.shape}.")
submission.make("fingerprint", test_pred)
