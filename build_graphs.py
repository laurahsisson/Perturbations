import os

import numpy as np
import torch
import torch_geometric as pyg
import tqdm
from hyperopt import hp
from hyperopt.pyll import scope
from ogb.utils import smiles2graph
from ray import train, tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.hyperopt import HyperOptSearch

import celltype
import data
import mrrmse
import submission


def to_torch(smiles):
    graph = smiles2graph(smiles)
    tensor_keys = ["edge_index", 'edge_feat', 'node_feat']
    for key in tensor_keys:
        graph[key] = torch.tensor(graph[key])
    return graph


def make_loader(data_x_entries, data_y, bsz):
    all_data = []
    for i, (ct, smiles) in enumerate(data_x_entries):
        y = torch.tensor(data_y[i]).unsqueeze(0)
        graph = to_torch(smiles)
        data = pyg.data.Data(ct=celltype.idx(ct), x=graph["node_feat"].float(), edge_index=graph["edge_index"],
                             edge_attr=graph["edge_feat"].float(), y=y.float(), smiles=smiles)
        all_data.append(data)
    return pyg.loader.DataLoader(all_data, batch_size=bsz)


def make_sequential(input_dim, hidden_dim, output_dim, dropout):
    return torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Dropout(p=dropout), torch.nn.Linear(hidden_dim, output_dim))


device = "cpu"


def num_node_features():
    return 9


def num_classes():
    return 18211


class MessagePassing(torch.nn.Module):

    def __init__(self, config):
        super(MessagePassing, self).__init__()

        # In order to tie the weights of all convolutions, the input is first
        # padded with zeros to reach embedding size.
        self.pad = torch.nn.ZeroPad2d(
            (0, config["mpnn_emb_size"] - num_node_features(), 0, 0))

        self.gins = []
        for _ in range(config["conv_layers"]):
            self.gins.append(pyg.nn.GINConv(make_sequential(config["mpnn_emb_size"], config["conv_hidden_dim"], config["mpnn_emb_size"], config["dropout"])).to(device))

        self.num_convs = config["num_convs"]

        # The pooling returns 2*emb_size, but MessagePassing is expected to return emb_size
        self.post_mp = make_sequential(
            2 * config["mpnn_emb_size"], config["conv_hidden_dim"], config["mpnn_emb_size"], config["dropout"])
        self.post_mp.to(device)

    def forward(self, db):
        x = self.pad(db.x)
        for gin in self.gins:
            for _ in range(self.num_convs):
                x = gin(x, db.edge_index)
                x = torch.nn.functional.relu(x)

        pooled = torch.cat([pyg.nn.pool.global_add_pool(
            x, db.batch), pyg.nn.pool.global_mean_pool(x, db.batch)], dim=1)

        return self.post_mp(pooled)


class EmbNN(torch.nn.Module):
    def __init__(self, config):
        super(EmbNN, self).__init__()
        self.ct_embed = torch.nn.Embedding(
            celltype.count(), config["ct_emb_size"])
        self.mpnn = MessagePassing(config)
        self.out = make_sequential(
            config["ct_emb_size"] + config["mpnn_emb_size"], config["combiner_hidden_dim"], num_classes(), config["dropout"])

    def forward(self, db):
        ct_res = self.ct_embed(db.ct)
        mpnn_res = self.mpnn(db)
        combined_res = torch.cat((ct_res, mpnn_res), dim=1)
        return self.out(combined_res)


train_x_entries, train_y = data.get_train(data.Include.All)
train_loader = make_loader(train_x_entries, train_y, 64)

eval_x_entries, eval_y = data.get_train(data.Include.Evaluation)
eval_loader = make_loader(eval_x_entries, eval_y, len(eval_x_entries))

test_x_entries = data.get_test()
# Placeholder values to build loader.
test_y = np.empty((len(test_x_entries), num_classes()))
test_loader = make_loader(test_x_entries, test_y, len(test_x_entries))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

epochs = 100
num_samples = 100


def train_model(config, input_data):
    def do_train_epoch():
        model.train()
        for batch_data in input_data["train_loader"]:
            optimizer.zero_grad()

            pred = model(batch_data)
            loss = loss_fn(pred, batch_data.y)

            loss.backward()
            optimizer.step()
        return loss

    def eval_score():
        assert len(input_data["eval_loader"]) == 1
        eval_batch = next(iter(input_data["eval_loader"]))
        with torch.no_grad():
            model.eval()
            return mrrmse.vectorized(model(eval_batch), eval_batch.y).item()

    model = EmbNN(config)
    config["parameters"] = count_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    loss_fn = torch.nn.MSELoss()

    best_loss = float('inf')
    for i in range(epochs):
        do_train_epoch()
        train.report({"mrrmse": eval_score()})
        config["completed"] = i
        if i % 5 == 0:
            # This saves the model to the trial directory
            torch.save(model.state_dict(), "./model.pt")


space = {
    "lr": hp.loguniform("lr", -10, -1),
    "dropout": hp.uniform("dropout", 0, 1),
    "ct_emb_size": scope.int(hp.qloguniform("ct_emb_size", 0, 3, 1)),
    "mpnn_emb_size": scope.int(hp.qloguniform("mpnn_emb_size", 0, 3, 1)),
    "combiner_hidden_dim": scope.int(hp.qloguniform("combiner_hidden_dim", 0, 3, 1)),
    "conv_hidden_dim": scope.int(hp.qloguniform("conv_hidden_dim", 0, 3, 1)),
    "num_convs": scope.int(hp.quniform("num_convs", 1, 10, 1)),
    "conv_layers": scope.int(hp.quniform("conv_layers", 1, 2, 1)),
}

metric = "mrrmse"
mode = "min"
hyperopt_search = HyperOptSearch(space, metric=metric, mode=mode)
scheduler = ASHAScheduler(metric=metric, grace_period=5, mode=mode, max_t=epochs)
input_data = {
    "train_loader": train_loader,
    "eval_loader": eval_loader,
}

tuner = tune.Tuner(
    tune.with_parameters(train_model, input_data=input_data),
    tune_config=tune.TuneConfig(
        num_samples=num_samples,
        search_alg=hyperopt_search,
        scheduler=scheduler
    ),
    run_config=train.RunConfig(
        failure_config=train.FailureConfig(fail_fast=False))
)
results = tuner.fit()

best_result = results.get_best_result(metric, mode=mode)
print(best_result.path)
print("CONFIG:", best_result.config)
print("METRICS:", best_result.metrics)

state_dict = torch.load(os.path.join(best_result.path, "model.pt"))
model = EmbNN(best_result.config)
model.load_state_dict(state_dict)

assert len(test_loader) == 1
test_batch = next(iter(test_loader))
with torch.no_grad():
    model.eval()
    pred = model(test_batch).numpy()
    submission.make("gnn", pred)
