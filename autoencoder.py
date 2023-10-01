import torch
import utils

import pandas as pd
import sklearn
import sklearn.model_selection
import mrrmse
import tqdm

df = pd.read_parquet("data/lincs.parquet")
xdf,ydf = df.iloc[:,:6], df.iloc[:,6:]

train,test = sklearn.model_selection.train_test_split(torch.from_numpy(ydf.to_numpy()))

class AutoEncoder(torch.nn.Module):
    def __init__(self,config):
        super(AutoEncoder,self).__init__()

        self.encoder = utils.make_sequential(config["input_dim"],config["hidden_dim"],config["latent_dim"],config["dropout"])
        self.decoder = utils.make_sequential(config["latent_dim"],config["hidden_dim"],config["input_dim"],config["dropout"])


    def forward(self,x):
        latent = self.encoder(x)
        latent = torch.nn.functional.relu(latent)
        return self.decoder(latent)

config = {
    "input_dim" : train.shape[1],
    "latent_dim": 256,
    "hidden_dim": 512,
    "dropout": .1,
}

def train_model():
    def do_train_epoch():
        model.train()
        for batch in loader:
            optimizer.zero_grad()
            loss = lossfn(model(batch),batch)
            loss.backward()
            optimizer.step()
        return loss

    def calc_test():
        model.eval()
        with torch.no_grad():
            return mrrmse.vectorized(model(test),test)

    loader = torch.utils.data.DataLoader(train, batch_size=128)
    model = AutoEncoder(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    lossfn = torch.nn.MSELoss()

    for i in tqdm.tqdm(range(100)):
        trnloss = do_train_epoch()
        tstloss = calc_test()
        print(trnloss,tstloss)

train_model()