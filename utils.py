import torch
import celltype
import numpy as np
import smiles

def make_sequential(input_dim, hidden_dim, output_dim, dropout):
    return torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Dropout(p=dropout), torch.nn.Linear(hidden_dim, output_dim))

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
