import torch

from torch_geometric.data import Batch
from torch.nn import CrossEntropyLoss
from typing import Union

from mpnn import MPNN
from gat import GATModel

def test(model: Union[MPNN, GATModel], pyg_batch: Batch):
    criterion = CrossEntropyLoss()
    model.eval()
    with torch.no_grad():
        if isinstance(model, MPNN):
            out = model(pyg_batch.x, pyg_batch.edge_attr, pyg_batch.u, pyg_batch.edge_index, pyg_batch.batch)
        else:  # GATModel
            out = model(pyg_batch.x, pyg_batch.edge_index)
        loss = criterion(out, pyg_batch.y)
        pred = out.argmax(dim=1)
        acc = (pred == pyg_batch.y).sum().item() / pyg_batch.y.size(0)
    return loss.item(), acc