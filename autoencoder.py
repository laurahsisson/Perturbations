import torch
import utils

import pandas as pd
import sklearn
import sklearn.model_selection
import mrrmse
import tqdm

# Much of this code is borrowed from 
# https://github.com/AntixK/PyTorch-VAE/tree/master
class AutoEncoder(torch.nn.Module):
    def __init__(self,target_dim,config):
        super(AutoEncoder,self).__init__()

        # These could be converted to a sequence of increase dimensions
        # like [32,64,128] .. 
        print(f"TARGETDIM={target_dim}")
        self.encoder = utils.make_sequential(target_dim,config["hidden_dim"],config["latent_dim"],config["dropout"])
        # and decoder would be reversed.
        self.decoder = utils.make_sequential(config["latent_dim"],config["hidden_dim"],target_dim,config["dropout"])

        # These would translate from the last dimension. But for now single layer is good enough.
        self.fc_mu = torch.nn.Linear(config["latent_dim"], config["latent_dim"])
        self.fc_var = torch.nn.Linear(config["latent_dim"], config["latent_dim"])

        # for the gaussian likelihood
        self.log_scale = torch.nn.Parameter(torch.Tensor([0.0]))

        self.kld_weight = config["kld_weight"]


    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def encode(self, x):
        latent = self.encoder(x)
        latent = torch.nn.functional.relu(latent)

        return self.fc_mu(latent), self.fc_var(latent)

    # Keep the encoder/decoder models private by providing helper functions.
    def latent(self,x):
        mu, log_var = self.encode(x)
        return {"z":self.reparameterize(mu, log_var), "mu": mu, "log_var":log_var}

    def decode(self,z):
        return self.decoder(z)

    def forward(self,x):
        latent = self.latent(x)
        return {"x_hat":self.decode(latent["z"]), "mu": latent["mu"], "log_var":latent["log_var"]}

    def loss_function(self,fwd,x):
        recons_loss = torch.nn.functional.mse_loss(fwd["x_hat"], x)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + fwd["log_var"] - fwd["mu"] ** 2 - fwd["log_var"].exp(), dim = 1), dim = 0)

        loss = recons_loss + self.kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

config = {
    "latent_dim": 256,
    "hidden_dim": 512,
    "dropout": .1,
    "kld_weight": 1,
}

def train_model():
    def do_train_epoch():
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            loss = model.loss_function(model(batch),batch)["loss"]
            loss.backward()
            optimizer.step()
        return loss

    def calc_test():
        model.eval()
        with torch.no_grad():
            pred = model(test)["x_hat"]
            return mrrmse.vectorized(pred,test)

    loader = torch.utils.data.DataLoader(train, batch_size=128)
    model = AutoEncoder(target_dim=978,config=config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    lossfn = torch.nn.MSELoss()

    for i in tqdm.tqdm(range(100)):
        trnloss = do_train_epoch()
        tstloss = calc_test()
        print(trnloss,tstloss)

if __name__ == "__main__":
    df = pd.read_parquet("data/lincs.parquet")
    xdf,ydf = df.iloc[:,:6], df.iloc[:,6:]

    train,test = sklearn.model_selection.train_test_split(torch.from_numpy(ydf.to_numpy()))
    train_model()