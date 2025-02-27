# What a great name lol

import autoencoder
import torch
import utils
import tqdm
import sklearn
import sklearn.model_selection
import mrrmse

loaded = torch.load("data/pretreatment.pt")
print(f"Keys = {loaded[0].keys()}")

column_uniques = {key:list({d[key] for d in loaded}) for key in loaded[0].keys()}
def get_idx(d,key):
    return column_uniques[key].index(d[key])

# Convert the columns we care about to idxs so that all values are numerical
data = [{"pert_iname":get_idx(d,"pert_iname"),"cell_id":get_idx(d,"cell_id"),"pre_treatment":d["pre_treatment"].float(),"post_treatment":d["post_treatment"].float()} for d in loaded]
train, test = sklearn.model_selection.train_test_split(data)
train_loader = torch.utils.data.DataLoader(train, batch_size=128)
test_loader = torch.utils.data.DataLoader(test, batch_size=128)

class Translator(torch.nn.Module):
    def __init__(self,config):
        super(Translator,self).__init__()
        # This will eventually be changed to a GNN
        self.pert_embed = torch.nn.Embedding(
            len(column_uniques["pert_iname"]), config["pert_emb_size"])

        # This needs to be able to handle out of dictionary
        self.cell_embed = torch.nn.Embedding(
            len(column_uniques["cell_id"]), config["cell_emb_size"])

        self.config = config
        input_dim = config["pert_emb_size"] + config["cell_emb_size"] + config["latent_dim"]
        self.translation = utils.make_sequential(input_dim,config["hidden_dim"],config["latent_dim"],config["dropout"])

    def forward(self,inp,z):
        pemb = self.pert_embed(inp["pert_iname"])
        cemb = self.cell_embed(inp["cell_id"])
        x = torch.cat((pemb, cemb, z), dim=1)
        return self.translation(x)

class RNVAE(torch.nn.Module):
    def __init__(self,target_dim,config):
        super(RNVAE,self).__init__()
        self.vae = autoencoder.AutoEncoder(target_dim=target_dim,config=config)
        self.translator = Translator(config)

    def forward(self,inp):
        assert inp["pre_treatment"].shape == inp["post_treatment"].shape
        latent = self.vae.latent(inp["pre_treatment"])
        z_prime = self.translator(inp,latent["z"])
        x_hat = self.vae.decode(z_prime)
        return {"x_hat":x_hat, "mu": latent["mu"], "log_var":latent["log_var"]}

    def loss_function(self,fwd,inp):
        return self.vae.loss_function(fwd,inp["post_treatment"])

config = {
    "pert_emb_size": 64,
    "cell_emb_size": 32,
    "latent_dim": 256,
    "hidden_dim": 512,
    "dropout": .1,
    "kld_weight": 1,
}

# inp = next(iter(loader))
# fwd = model(inp)
# print(model.loss_function(fwd,inp))

def train_model():
    model = RNVAE(target_dim=978,config=config)

    def do_train_epoch():
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model.loss_function(model(batch),batch)["loss"]
            loss.backward()
            optimizer.step()
        return loss

    def calc_test():
        model.eval()
        with torch.no_grad():
            batch = next(iter(test_loader))
            pred = model(batch)["x_hat"]
            return mrrmse.vectorized(pred,batch["post_treatment"])

    model = RNVAE(target_dim=978,config=config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    for i in tqdm.tqdm(range(100)):
        trnloss = do_train_epoch()
        tstloss = calc_test()
        print(trnloss,tstloss)


train_model()