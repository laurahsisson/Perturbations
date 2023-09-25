import torch
import sklearn
import sklearn.decomposition

bsz = 1000
dim = 128
nc = 64
k = torch.rand(bsz,dim)
print(k.shape)
decomposer = sklearn.decomposition.TruncatedSVD(n_components=nc).fit(k)
print(torch.from_numpy(decomposer.transform(k)).shape)
print(decomposer.components_.shape)
print(decomposer.n_features_in_)