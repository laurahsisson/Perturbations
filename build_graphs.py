import torch
import torch_geometric as pyg
import tqdm
from ogb.utils import smiles2graph

import celltype
import data
import mrrmse


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


def make_sequential(input_dim, hidden_dim, output_dim, dropout=0):
    return torch.nn.Sequential(torch.nn.Linear(input_dim, hidden_dim), torch.nn.ReLU(), torch.nn.Dropout(p=dropout), torch.nn.Linear(hidden_dim, output_dim))


device = "cpu"


def num_node_features():
    return 9


def num_classes():
    return 18211


class MessagePassing(torch.nn.Module):

    def __init__(self, embedding_size, num_convs=3, hidden_dim=128):
        super(MessagePassing, self).__init__()

        # In order to tie the weights of all convolutions, the input is first
        # padded with zeros to reach embedding size.
        self.pad = torch.nn.ZeroPad2d(
            (0, embedding_size - num_node_features(), 0, 0))

        self.gcn = pyg.nn.GINConv(make_sequential(
            embedding_size, hidden_dim, embedding_size))
        self.gcn.to(device)
        self.num_convs = num_convs

        # The pooling returns 2*emb_size, but MessagePassing is expected to return emb_size
        self.post_mp = make_sequential(
            2 * embedding_size, hidden_dim, embedding_size)
        self.post_mp.to(device)

    def forward(self, db):
        x = self.pad(db.x)
        for _ in range(self.num_convs):
            x = self.gcn(x, db.edge_index)

        pooled = torch.cat([pyg.nn.pool.global_add_pool(
            x, db.batch), pyg.nn.pool.global_mean_pool(x, db.batch)], dim=1)

        return self.post_mp(pooled)


class EmbNN(torch.nn.Module):
    def __init__(self, ct_emb_size=2, mpnn_emb_size=64, hidden_dim=32):
        super(EmbNN, self).__init__()
        self.ct_embed = torch.nn.Embedding(celltype.count(), ct_emb_size)
        self.mpnn = MessagePassing(mpnn_emb_size)
        self.out = make_sequential(
            ct_emb_size+mpnn_emb_size, hidden_dim, num_classes())

    def forward(self, db):
        ct_res = self.ct_embed(db.ct)
        mpnn_res = self.mpnn(db)
        combined_res = torch.cat((ct_res, mpnn_res), dim=1)
        return self.out(combined_res)


train_x_entries, train_y = data.get_train(data.Include.All)
train_loader = make_loader(train_x_entries, train_y, 64)

eval_x_entries, eval_y = data.get_train(data.Include.Evaluation)
eval_loader = make_loader(eval_x_entries, eval_y, len(eval_x_entries))


def train_model(config, input_data):
    def do_train_epoch():
        model.train()
        for batch_data in train_loader:
            optimizer.zero_grad()

            pred = model(batch_data)
            loss = loss_fn(pred, batch_data.y)

            loss.backward()
            optimizer.step()
        return loss

    model = EmbNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=.00001)
    loss_fn = torch.nn.MSELoss()

    best_loss = float('inf')
    eps = 1e-6
    for _ in tqdm.tqdm(range(100000)):
        loss = do_train_epoch()
        print(loss)
        if loss + eps < best_loss:
            best_loss = loss
        else:
            break

    eval_batch = next(iter(eval_loader))
    with torch.no_grad():
        model.eval()
        print(mrrmse.vectorized(model(eval_batch), eval_batch.y))
