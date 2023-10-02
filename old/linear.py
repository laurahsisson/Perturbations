import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rdkit
import rdkit.Chem.rdFingerprintGenerator
import sklearn
import sklearn.decomposition
import sklearn.kernel_ridge
import sklearn.linear_model
import sklearn.metrics
import torch
import tqdm

import celltype
import data
import mrrmse
import smiles
import submission

# https://github.com/rdkit/rdkit/discussions/3863


def to_numpy(fp):
    return np.frombuffer(bytes(fp.ToBitString(), 'utf-8'), 'u1') - ord('0')


def convert_to_input(fpfn, x_entry):
    ct, smls = x_entry
    ct = celltype.one_hot(ct)
    mfp = fpfn(smls)
    return mfp  # np.concatenate([ct, mfp])


def weight(x_entry):
    ct, smiles = x_entry
    if celltype.is_evaluation(ct):
        return 1000
    return 1


def model(fname, fpfn):
    train_x_entries, train_y = data.get_train(data.Include.All)
    eval_x_entries, eval_y = data.get_train(data.Include.Evaluation)

    train_x = np.stack([convert_to_input(fpfn, x_entry)
                       for x_entry in train_x_entries])
    train_weights = np.stack([weight(x_entry) for x_entry in train_x_entries])
    train_y = np.stack(train_y)
    decomposer = sklearn.decomposition.TruncatedSVD(
        n_components=5000).fit(train_y)

    eval_x = np.stack([convert_to_input(fpfn, x_entry)
                      for x_entry in eval_x_entries])
    eval_y = np.stack(eval_y)

    print("Training model.")
    model = sklearn.kernel_ridge.KernelRidge(kernel='linear').fit(
        train_x, decomposer.transform(train_y), sample_weight=train_weights)

    train_pred = decomposer.inverse_transform(model.predict(train_x))
    eval_pred = decomposer.inverse_transform(model.predict(eval_x))
    print("Eval", mrrmse.vectorized(
        torch.tensor(eval_pred), torch.tensor(eval_y)))

    test_x_entries = data.get_test()
    test_x = np.stack([convert_to_input(fpfn, x_entry)
                      for x_entry in test_x_entries])
    test_pred = decomposer.inverse_transform(model.predict(test_x))
    print(f"Made predictions of shape {test_pred.shape}.")
    bins = np.linspace(-10, 10, 100)

    plt.hist(eval_pred.flatten(), bins, alpha=0.5, label='pred')
    plt.hist(eval_y.flatten(), bins, alpha=0.5, label='actual')
    plt.legend(loc='upper right')
    plt.show()
    # submission.make(fname, test_pred)


def fingerprint():
    mfpgen = rdkit.Chem.rdFingerprintGenerator.GetMorganGenerator(radius=4,
                                                                  fpSize=128)
    # Closure city

    def get_fingerprint(smls):
        mol = rdkit.Chem.MolFromSmiles(smls)
        fp = mfpgen.GetFingerprint(mol)
        return to_numpy(fp)

    model("fingerprint", get_fingerprint)


def onehot():
    model("onehot", smiles.one_hot)


if __name__ == "__main__":
    fingerprint()
