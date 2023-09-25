import torch
import sklearn
import sklearn.decomposition
import data
import numpy as np


def to_tensor(lst):
    return torch.from_numpy(np.array(lst)).float().squeeze()


train_x_entries, train_y = data.get_train(data.Include.NonEvaluation)


bsz = 1000
dim = 128
nc = 64
k = torch.rand(bsz, dim)
print(k.shape)
dec1 = sklearn.decomposition.TruncatedSVD(n_components=nc).fit(k)
print(dec1.transform(k).shape)
print(dec1.components_.shape)
print(dec1.n_features_in_)

train_y = to_tensor(train_y)


print(train_y.shape)

print(k.dtype,)
print(train_y.dtype)

tz = torch.rand(train_y.shape)
dec2 = sklearn.decomposition.TruncatedSVD(n_components=30).fit(tz)
print("TZ", tz.shape)
print(len(train_y))
print(dec2.transform(train_y).shape)
print(dec2.transform(tz).shape)
