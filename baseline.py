# load pt data
# load kaggle data
# combine and merge cell_types and smiles
# model is just single layer perceptrons across the board.

import torch
import sklearn
import sklearn.model_selection
import submission

print("Loading data")
lincs = torch.load("data/lincs_pretreatment.pt")
lincs_uniques = {
    key: list({d[key]
               for d in lincs})
    for key in ["cell_id", "canonical_smiles"]
}

kaggle = torch.load("data/kaggle_pretreatment.pt")
kaggle_uniques = {
    key: list({d[key]
               for d in kaggle})
    for key in ["cell_type", "SMILES"]
}

test = torch.load("data/test_pretreatment.pt")

cell_types = list(set(lincs_uniques["cell_id"] + kaggle_uniques["cell_type"]))
smiles_types = list(
    set(lincs_uniques["canonical_smiles"] + kaggle_uniques["SMILES"]))

def make_data(df,smiles_col_name,cell_col_name):
    def trch(t):
        return torch.from_numpy(t).float()
    
    def make_dp(d):
        dp = {
            "smiles": smiles_types.index(d[smiles_col_name]),
            "cell": cell_types.index(d[cell_col_name]),
            "pre_treatment": trch(d["pre_treatment"]),
        }
        if "transcriptome" in d:
            dp["transcriptome"] = trch(d["transcriptome"])
        if "post_treatment" in d:
            dp["post_treatment"] = trch(d["post_treatment"])
        return dp

    return [make_dp(d) for d in df]

print("Building datasets")
lincs_data = make_data(lincs,"canonical_smiles","cell_id")
kaggle_data = make_data(kaggle,"SMILES","cell_type")
# We really should do cross-validation.
kaggle_train, kaggle_eval = sklearn.model_selection.train_test_split(kaggle_data)

eval_loader = torch.utils.data.DataLoader(kaggle_eval, batch_size=len(kaggle_eval))


class Translator(torch.nn.Module):
    def __init__(self,target_dim,config):
        super(Translator,self).__init__()
        self.pert_embed = torch.nn.Embedding(
            len(smiles_types), config["smiles_emb_size"])

        # This needs to be able to handle out of dictionary
        self.cell_embed = torch.nn.Embedding(
            len(cell_types), config["cell_emb_size"])

        self.config = config
        input_dim = config["smiles_emb_size"] + config["cell_emb_size"] + target_dim
        self.translation = torch.nn.Linear(input_dim, target_dim)

    def forward(self,inp):
        pemb = self.pert_embed(inp["smiles"])
        cemb = self.cell_embed(inp["cell"])
        x = torch.cat((pemb, cemb, inp["pre_treatment"]), dim=1)
        return self.translation(x)

class Imputer(torch.nn.Module):
    def __init__(self,target_dim,impute_dim):
        super(Imputer,self).__init__()
        self.impute = torch.nn.Linear(target_dim, impute_dim)

    def forward(self,landmark):
        return self.impute(landmark)

config = {
    "smiles_emb_size": 64,
    "cell_emb_size": 32,
}



def train_translator():
    print("Training translator")
    # Can use both lincs data and kaggle data.
    train_data = lincs_data + kaggle_train
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128)

    model = Translator(918,config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.MSELoss()
    best_loss = float('inf')
    for epoch in range(100):
        for batch in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch),batch["post_treatment"])
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            batch = next(iter(eval_loader))
            loss = loss_fn(model(batch),batch["post_treatment"])
            print(epoch,loss)
            if loss < best_loss:
                best_loss = loss
            else:
                break
    return model

# Have to pass in the baseline
def train_imputer(translator):
    print("Training imputer")
    # Can only use kaggle data.
    train_loader = torch.utils.data.DataLoader(kaggle_train, batch_size=128)
    model = Imputer(918,18211)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.MSELoss()
    best_loss = float('inf')
    for epoch in range(100):
        for batch in train_loader:
            optimizer.zero_grad()
            landmark = translator(batch)
            loss = loss_fn(model(landmark),batch["transcriptome"])
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            batch = next(iter(eval_loader))
            landmark = translator(batch)
            loss = loss_fn(model(landmark),batch["transcriptome"])
            print(epoch,loss)
            if loss < best_loss:
                best_loss = loss
            else:
                break
    return model

translator = train_translator()
imputer = train_imputer(translator)

print("Preparing submission data.")
test_data = make_data(test,"SMILES","cell_type")
test_loader = torch.utils.data.DataLoader(test_data, batch_size=len(test_data))

with torch.no_grad():
    pred = imputer(translator(next(iter(test_loader)))).numpy()
    submission.make("baseline", pred)