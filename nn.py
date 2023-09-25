import pickle
import sklearn.metrics
import matplotlib.pyplot as plt
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch
from ray import train, tune
import rdkit
import rdkit.Chem.rdFingerprintGenerator

import os
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
from hyperopt import hp

os.environ["RAY_DEDUP_LOGS"] = "0"


# https://github.com/rdkit/rdkit/discussions/3863


def to_numpy(fp):
    return np.frombuffer(bytes(fp.ToBitString(), 'utf-8'), 'u1') - ord('0')


def get_fingerprint(smls):
    mol = rdkit.Chem.MolFromSmiles(smls)
    fp = mfpgen.GetFingerprint(mol)
    return to_numpy(fp)


def to_tensor(lst):
    return torch.tensor(np.array(lst), dtype=torch.float32)


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


def split_entries(x_entries):
    weights = torch.tensor([weight(x_entry) for x_entry in x_entries])
    x_ct = to_tensor([celltype.one_hot(ct) for ct, smls in x_entries])
    x_smls = to_tensor([smiles.one_hot(smls)
                        for ct, smls in x_entries])
    return weights, x_ct, x_smls


train_x_entries, train_y = data.get_train(data.Include.NonEvaluation)
train_weights, train_x_ct, train_x_smls = split_entries(train_x_entries)
train_y = to_tensor(train_y)

eval_x_entries, eval_y = data.get_train(data.Include.Evaluation)
_, eval_x_ct, eval_x_smls = split_entries(eval_x_entries)
eval_y = to_tensor(eval_y)


class EmbNN(torch.nn.Module):
    num_categories1 = 6
    num_categories2 = 146

    def __init__(self, config):
        super(EmbNN, self).__init__()
        # Embedding layer for categorical feature 1
        self.embedding1 = torch.nn.Embedding(
            self.num_categories1, int(config["ct_emb_size"]))
        # Embedding layer for categorical feature 2
        self.embedding2 = torch.nn.Embedding(
            self.num_categories2, int(config["smls_emb_size"]))
        self.out = torch.nn.Sequential(torch.nn.Linear(int(config["ct_emb_size"] + config["smls_emb_size"]), int(config["embed_size"])), torch.nn.ReLU(
        ), torch.nn.Dropout(p=0), torch.nn.Linear(int(config["embed_size"]), config["out_dim"]))

    def forward(self, cts, smlss):
        ct_idx = torch.argmax(cts, dim=1)
        smls_idx = torch.argmax(smlss, dim=1)
        # Embed categorical feature 1
        cat_embed1 = self.embedding1(ct_idx.long())
        # Embed categorical feature 2
        cat_embed2 = self.embedding2(smls_idx.long())
        x = torch.cat((cat_embed1.view(ct_idx.size(0), -1), cat_embed2.view(
            smls_idx.size(0), -1)), dim=1)  # Concatenate with continuous input
        return self.out(x)


def weighted_mse(y_pred, y_true, weights):
    squared_errors = torch.square(y_pred - y_true)
    weighted_squared_errors = squared_errors * weights.unsqueeze(1)
    loss = torch.mean(weighted_squared_errors)
    return loss


epochs = 1000
num_samples = 100


def train_model(config, input_data):
    _train_y = input_data["train_y"]

    config["out_dim"] = 18211
    has_decomposer = config["reduction"]["type"] == "tsvd"
    if has_decomposer:
        config["out_dim"] = int(config["reduction"]["components"])
        decomposer = sklearn.decomposition.TruncatedSVD(n_components=config["out_dim"]).fit(input_data["train_y"])
        _train_y = torch.from_numpy(decomposer.transform(input_data["train_y"]))
        torch.save(decomposer, "./decomposer.pkl")
        

    model = EmbNN(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    model.train()
    for i in range(epochs):
        optimizer.zero_grad()
        train_pred = model(input_data["train_x_ct"],
                           input_data["train_x_smls"])
        if train_pred.shape != _train_y.shape:
            print("DDD", config["out_dim"], input_data["train_y"].shape, train_pred.shape, _train_y.shape, decomposer.components_.shape)
            raise RuntimeError("")
        loss = weighted_mse(
            train_pred, _train_y, input_data["train_weights"])

        eval_pred = model(
            input_data["eval_x_ct"], input_data["eval_x_smls"]).detach()
        if has_decomposer:
            eval_pred = torch.tensor(
                decomposer.inverse_transform(eval_pred.numpy()))
        score = mrrmse.vectorized(eval_pred, input_data["eval_y"]).item()
        train.report({"mrrmse": score})
        loss.backward()
        optimizer.step()

        if i % 5 == 0:
            # This saves the model to the trial directory
            torch.save(model.state_dict(), "./model.pt")


space = {
    "lr": hp.loguniform("lr", -10, -1),
    "ct_emb_size": hp.quniform("ct_emb_size", 2, 100, 1),
    "smls_emb_size": hp.quniform("smls_emb_size", 2, 100, 1),
    "embed_size": hp.quniform("embed_size", 2, 100, 1),
    "reduction": hp.choice('reduction', [
        {'type': 'none'},
        {'type': 'tsvd', 'components': hp.qloguniform(
            'components', 0, 8, 1)},
    ]),
}

metric = "mrrmse"
mode = "min"
hyperopt_search = HyperOptSearch(space, metric=metric, mode=mode)
scheduler = ASHAScheduler(metric=metric, mode=mode, max_t=epochs)
input_data = {
    "train_x_ct": train_x_ct,
    "train_x_smls": train_x_smls,
    "train_y": train_y,
    "train_weights": train_weights,
    "eval_x_ct": eval_x_ct,
    "eval_x_smls": eval_x_smls,
    "eval_y": eval_y,
}
tuner = tune.Tuner(
    tune.with_parameters(train_model, input_data=input_data),
    tune_config=tune.TuneConfig(
        num_samples=num_samples,
        search_alg=hyperopt_search,
        scheduler=scheduler
    ),
    run_config=train.RunConfig(
        failure_config=train.FailureConfig(fail_fast=True))
)
results = tuner.fit()

best_result = results.get_best_result(metric, mode=mode)
state_dict = torch.load(os.path.join(best_result.path, "model.pt"))

print(best_result.config, best_result.metrics)
model = EmbNN(best_result.config)
model.load_state_dict(state_dict)


test_x_entries = data.get_test()
_, test_x_ct, test_x_smls = split_entries(test_x_entries)

# model.eval()
# with torch.no_grad():
#     pred = model(test_x_ct, test_x_smls).numpy()
#     if best_result.config["reduction"]["type"] == "tsvd":
#         decomposer = torch.load(os.path.join(
#             best_result.path, "decomposer.pkl"))
#         pred = decomposer.inverse_transform(pred)

#     submission.make("dictionary", pred)
