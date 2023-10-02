# What a great name lol

import autoencoder
import torch
import utils

loaded = torch.load("data/pretreatment.pt")
print(f"Keys = {loaded[0].keys()}")

column_uniques = {key:list({d[key] for d in loaded}) for key in loaded[0].keys()}
def get_idx(d,key):
    return column_uniques[key].index(d[key])

# Convert the columns we care about to idxs so that all values are numerical
data = [{"pert_iname":get_idx(d,"pert_iname"),"cell_id":get_idx(d,"cell_id"),"pre_treatment":d["pre_treatment"],"post_treatment":d["post_treatment"]} for d in loaded]
loader = torch.utils.data.DataLoader(data, batch_size=128)
print(next(iter(loader)))


class Translator(torch.nn.Module):
    def __init__(self,config):
        super(Translator,self).__init__()
        # This will eventually be changed to a GNN
        self.pert_embed = torch.nn.Embedding(
            len(column_uniques["pert_iname"]), config["pert_emb_size"])

        # This needs to be able to handle out of dictionary
        self.cell_embed = torch.nn.Embedding(
            len(column_uniques["cell_id"]), config["cell_emb_size"])

        input_dim = config["pert_emb_size"] + config["cell_emb_size"] + config["latent_dim"]
        self.translation = utils.make_sequential(input_dim,config["hidden_dim"],config["latent_dim"],config["dropout"])

    def forward(self,inp,z):
        pemb = self.pert_embed(inp["pert_iname"])
        cemb = self.pert_embed(inp["cell_id"])
        x = torch.cat((pemb, cemb, z), dim=1)
        return self.translation(x)

class RNVAE(torch.nn.Module):
    def __init__(self,target_dim,config):
        super(RNVAE,self).__init__()
        self.vae = autoencoder.AutoEncoder(config)
        self.translator = Translator(target_dim,config)

    def forward(self,inp):
        latent = self.vae.latent(inp["pre_treatment"])
        z_prime = self.translator(inp,latent["z"])
        x_hat = self.vae.decode(z_prime)
        return {"x_hat":x_hat, "mu": latent["mu"], "log_var":latent["log_var"]}

    def loss_function(self,fwd,inp):
        return self.vae.loss_function(fwd,inp["post_treatment"])




        

