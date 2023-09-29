import torch
import utils

import data

class AutoEncoder(torch.nn.Module):
    def __init__(self,config):
        super(AutoEncoder,self).__init__()

        self.encoder = utils.make_sequential(config["input_dim"],config["hidden_dim"],config["latent_dim"],config["dropout"])
        self.decoder = utils.make_sequential(config["latent_dim"],config["hidden_dim"],config["input_dim"],config["dropout"])


    def forward(self,x):
        latent = self.encoder(x)
        latent = torch.nn.functional.relu(latent)
        return self.decoder(latent)


train_x_entries, train_y = data.get_train(data.Include.NonEvaluation)
train_weights, train_x_ct, train_x_smls = utils.split_entries(train_x_entries)
train_y = utils.to_tensor(train_y)

eval_x_entries, eval_y = data.get_train(data.Include.Evaluation)
_, eval_x_ct, eval_x_smls = utils.split_entries(eval_x_entries)
eval_y = utils.to_tensor(eval_y)


config = {
    "input_dim" : train_y.shape[1],
    "latent_dim": 256,
    "hidden_dim": 512,
    "dropout": .1,
}

model = AutoEncoder(config)

print(model(train_y).shape)