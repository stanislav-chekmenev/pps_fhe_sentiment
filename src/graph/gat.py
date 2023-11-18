import torch

from torch_geometric.nn import GATConv
from torch_geometric.data import  Batch

class GATModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_heads):
        super(GATModel, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=0.5)
        self.conv2 = GATConv(hidden_channels * num_heads, out_channels, heads=1, concat=False, dropout=0.5)
        self.dropout =torch.nn.Dropout(0.5)
        self.elu = torch.nn.ELU()

    def forward(self, batch: Batch):
        x, edge_index = batch.x, batch.edge_index
        x = self.dropout(x)
        x = self.conv1(x, edge_index)
        x = self.elu(x)
        x = self.dropout(x)
        return self.conv2(x, edge_index), x