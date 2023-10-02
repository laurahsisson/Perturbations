import numpy as np
import sklearn
import sklearn.decomposition
import torch

import celltype
import data
import smiles


def to_tensor(lst):
    return torch.tensor(np.array(lst), dtype=torch.float32)


def weight(x_entry):
    ct, smiles = x_entry
    if celltype.is_evaluation(ct):
        return 1000
    return 1


def split_entries(x_entries):
    weights = torch.tensor([weight(x_entry) for x_entry in x_entries])
    x_ct = to_tensor([celltype.one_hot(ct) for ct, smls in x_entries])
    x_smls = to_tensor([smiles.one_hot(smls)
                        for ct, smls in x_entries])
    return weights, x_ct, x_smls


train_x_entries, train_y = data.get_train(data.Include.NonEvaluation)
train_weights, train_x_ct, train_x_smls = split_entries(train_x_entries)


train_x_ct = train_x_ct[:2, :]
embedding1 = torch.nn.Embedding(6, 10)
ct_idx = torch.argmax(train_x_ct, dim=1)
print(ct_idx)
ct_embed = embedding1(ct_idx.long())
print(ct_embed, ct_embed.shape)
ct_view = ct_embed.view(ct_idx.size(0), -1)
assert torch.equal(ct_embed, ct_view)
