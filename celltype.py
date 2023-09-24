import pandas as pd
import torch

all_cell_types = [
    'T cells CD4+', 'T cells CD8+', 'Myeloid cells', 'B cells',
    'T regulatory cells', 'NK cells'
]
evaluation_cell_types = ['Myeloid cells', 'B cells']

df = pd.read_parquet("data/de_train.parquet")


def one_hot(cell_type):
    idx = torch.tensor(all_cell_types.index(cell_type))
    return torch.nn.functional.one_hot(idx, num_classes=len(all_cell_types))


def is_evaluation(cell_type):
    return cell_type in evaluation_cell_types


if __name__ == "__main__":
    for index, row in df.iterrows():
        print(one_hot(row["cell_type"]))
