import torch

from torch_geometric.nn import global_mean_pool
from torch_geometric.data import Batch
from torch_scatter import scatter_mean
from typing import Tuple


class EncoderModel(torch.nn.Module):
    def __init__(self, input_node_dim: int, input_edge_dim: int, input_global_dim: int,
                 emb_node_features: int, emb_edge_features: int, emb_global_features: int):
        super().__init__()
        self.node_mlp = torch.nn.Sequential(
            torch.nn.Linear(input_node_dim, emb_node_features),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_node_features, emb_node_features)
        )
        
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(input_edge_dim, emb_edge_features),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_edge_features, emb_edge_features)
        )
        
        self.global_mlp = torch.nn.Sequential(
            torch.nn.Linear(input_global_dim, emb_global_features),
            torch.nn.ReLU(),
            torch.nn.Linear(emb_global_features, emb_global_features)
        )

    def forward(self, node_attr, edge_attr, edge_index, u, batch):
        batch
        return self.node_mlp(node_attr), \
            self.edge_mlp(edge_attr), \
            self.global_mlp(u)


class EdgeModel(torch.nn.Module):
    def __init__(self, dim_node_features: int, dim_edge_features: int, dim_global_features: int,
                  hidden_channels: int) -> None:
        super().__init__()
        torch.manual_seed(42)
        num_features = dim_node_features * 2 + dim_edge_features + dim_global_features
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(num_features, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_channels, dim_edge_features),
            torch.nn.ReLU()
        )

    def forward(self, src: torch.Tensor, dest: torch.Tensor, edge_attr: torch.Tensor,
                 u: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        out = torch.cat([src, dest, edge_attr, u[batch]], 1)
        return self.edge_mlp(out)


class NodeModel(torch.nn.Module):
    def __init__(self, dim_node_features: int, dim_edge_features: int, dim_global_features: int,
                  hidden_channels: int) -> None:
        super().__init__()
        torch.manual_seed(42)
        num_features = dim_node_features + dim_edge_features + dim_global_features
        self.node_mlp = torch.nn.Sequential(
            torch.nn.Linear(num_features, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_channels, dim_node_features),
            torch.nn.ReLU()
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor,
                 u: torch.Tensor, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dest_node_idx = edge_index[1]
        edge_out_bar = scatter_mean(src=edge_attr, index=dest_node_idx, dim=0)
        out = torch.cat([x, edge_out_bar, u[batch]], 1)
        return self.node_mlp(out), edge_out_bar


class GlobalModel(torch.nn.Module):
    def __init__(self, dim_node_features: int, dim_edge_features: int, dim_global_features: int,
                  hidden_channels: int) -> None:
        super().__init__()
        torch.manual_seed(42)
        num_features = dim_node_features + dim_edge_features + dim_global_features
        self.global_mlp = torch.nn.Sequential(
            torch.nn.Linear(num_features, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_channels, dim_global_features),
            torch.nn.ReLU()
        )

    def forward(self, node_attr_prime: torch.Tensor, edge_out_bar: torch.Tensor,
                 u: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        node_attr_bar = scatter_mean(node_attr_prime, batch, dim=0)
        edge_attr_bar = scatter_mean(edge_out_bar, batch, dim=0)
        out = torch.cat([node_attr_bar, edge_attr_bar, u], dim=1)
        return self.global_mlp(out)


class MPNN(torch.nn.Module):

    def __init__(self, input_node_dim: int, input_edge_dim: int, input_global_dim: int,
                 dim_node_features: int, dim_edge_features: int, dim_global_features: int,
                 hidden_channels: int, num_classes: int, num_passes: int
                 ) -> None:
        super().__init__()
        torch.manual_seed(42)
        self.encoder = EncoderModel(input_node_dim, input_edge_dim, input_global_dim,
                 dim_node_features, dim_edge_features, dim_global_features)
        self.edge_model = EdgeModel(dim_node_features, dim_edge_features, dim_global_features, hidden_channels)
        self.node_model = NodeModel(dim_node_features, dim_edge_features, dim_global_features, hidden_channels)
        self.global_model = GlobalModel(dim_node_features, dim_edge_features, dim_global_features, hidden_channels)
        dim_all_features = dim_node_features + dim_edge_features + dim_global_features
        self.lin = torch.nn.Linear(dim_all_features, num_classes)
        self.num_passes = num_passes
        self.reset_parameters()

    def reset_parameters(self):
        for item in [self.node_model, self.edge_model, self.global_model]:
            if hasattr(item, 'reset_parameters'):
                item.reset_parameters()

    def forward(self, batch: Batch) -> torch.Tensor:
        node_attr, edge_attr, u, edge_index, batch = batch.x, batch.edge_attr, batch.u, batch.edge_index, batch.batch
        # 1. Encoding step
        node_attr, edge_attr, u = self.encoder(node_attr, edge_attr, edge_index, u, batch)

        # 2. Perform MPNN updates
        for _ in range(self.num_passes):
            src_node_idx = edge_index[0]
            dest_node_idx = edge_index[1]

            edge_attr = self.edge_model(
                node_attr[src_node_idx], node_attr[dest_node_idx], edge_attr,
                u, batch[dest_node_idx]
            )

            node_attr, edge_out_bar = self.node_model(
                node_attr, edge_index, edge_attr, u, batch
            )

            u = self.global_model(node_attr, edge_out_bar, u, batch)

        # 3. Readout layer
        graph_attr = torch.cat([node_attr, edge_out_bar, u[batch]], dim=1)
        return self.lin(graph_attr), node_attr


    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(\n'
                f'  edge_model={self.edge_model},\n'
                f'  node_model={self.node_model},\n'
                f'  global_model={self.global_model}\n'
                f')')