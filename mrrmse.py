import torch

rows = 110
cols = 3000
actual = torch.rand(rows,cols,dtype=torch.float64)

def vectorized(pred,actual):
    diff = torch.square(pred - actual)
    col_sum = torch.sum(diff,axis=1)/cols
    row_sum = torch.sum(torch.sqrt(col_sum))/rows
    return row_sum


def naive(pred,actual):
    t = 0
    for i in range(rows):
        s = 0
        for j in range(cols):
            s += torch.square(pred[i,j] - actual[i,j])
        t += torch.sqrt(s/cols)
    return t/rows


def batched(pred,actual,bsz,lsfn):
    ac_batch = torch.split(actual,bsz)
    pred_batch = torch.split(pred,bsz)
    assert len(ac_batch) == len(pred_batch)
    vals = []
    for b in range(len(ac_batch)):
        # Batches can be different sizes
        bsz = pred_batch[b].shape[0]
        vals.append(bsz*lsfn(pred_batch[b],ac_batch[b]))
    vals = torch.stack(vals)
    return torch.sum(vals) / pred.shape[0]

def rmse(pred,actual):
    return torch.sqrt(torch.nn.functional.mse_loss(pred,actual))

if __name__ == "__main__":
    for i in range(10000):
        pred1 = torch.rand(rows,cols,dtype=torch.float64)
        pred2 = torch.rand(rows,cols,dtype=torch.float64)
        print(i)
        if rmse(pred1,actual) < rmse(pred2,actual):
            if not vectorized(pred1,actual) < vectorized(pred2,actual):
                error = vectorized(pred1,actual) - vectorized(pred2,actual)
                print(error)
                exit()
        else:
            if not vectorized(pred1,actual) > vectorized(pred2,actual):
                error = vectorized(pred1,actual) - vectorized(pred2,actual)
                print(error)
                exit()

