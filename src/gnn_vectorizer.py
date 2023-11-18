import networkx as nx
import pickle
import torch
import numpy as np

from torch_geometric.data import Data, Batch
from sklearn.metrics.pairwise import cosine_similarity

from src.graph.mpnn import MPNN
from src.transformer_vectorizer import TransformerVectorizer


class GNNVectorizer():
    EPOCHS = 28
    INPUT_NODE_DIM = 768
    INPUT_EDGE_DIM = 1
    INPUT_GLOBAL_DIM = 1
    DIM_NODE_FEATURES = 128
    DIM_EDGE_FEATURES = 32
    DIM_GLOBAL_FEATURES = 32
    HIDDEN_CHANNELS = 32
    NUM_CLASSES = 3
    NUM_PASSES = 1

    def __init__(self, model_path="models/best_gnn.pth", 
                 subgraphs_folder="data/preprocessed_data/train_subgraphs.pkl"):
        self.gnn = MPNN(GNNVectorizer.INPUT_NODE_DIM, 
                        GNNVectorizer.INPUT_EDGE_DIM, 
                        GNNVectorizer.INPUT_GLOBAL_DIM,
                        GNNVectorizer.DIM_NODE_FEATURES, 
                        GNNVectorizer.DIM_EDGE_FEATURES, 
                        GNNVectorizer.DIM_GLOBAL_FEATURES,
                        GNNVectorizer.HIDDEN_CHANNELS, 
                        GNNVectorizer.NUM_CLASSES, 
                        GNNVectorizer.NUM_PASSES
                        )
        self.gnn.load_state_dict(torch.load(model_path))
        self.gnn.eval()
        self.transformer_vectorizer = TransformerVectorizer()
        self.subgraphs_folder = subgraphs_folder

    def text_to_tensor(self, texts: list):
        # Vectorize text
        encoding = self.transformer_vectorizer.transform(texts)

        # Load a random subgraph
        with open(self.subgraphs_folder, 'rb') as file:
            subgraphs = pickle.load(file)
        subgraph = subgraphs[np.random.randint(0, len(subgraphs))]

        # Add new node with encoding as features
        new_node_idx = max(subgraph.nodes) + 1
        subgraph.add_node(new_node_idx, features=encoding[0])

        # Add edges based on cosine similarity
        for i in subgraph.nodes:
            if i != new_node_idx:
                similarity = cosine_similarity(
                    [subgraph.nodes[i]['features']], 
                    [subgraph.nodes[new_node_idx]['features']]
                )[0][0]
                subgraph.add_edge(i, new_node_idx, weight=similarity)

        # Convert to PyG Data
        pyg_data = self.to_pyg_data(subgraph, 1)
        pyg_batch = Batch.from_data_list([pyg_data])

        # Forward pass through the model
        _, gnn_encoding = self.gnn(pyg_batch)

        # Clear memory
        return gnn_encoding[-1]
    
    def transform(self, texts: list):
        return self.text_to_tensor(texts).detach().cpu().numpy().reshape(1, -1)

    def to_pyg_data(self, G: nx.Graph, dim_global_features: int) -> Data:
        edge_index = torch.tensor(list(G.edges)).t().contiguous()
        edge_attr = torch.tensor(np.array([G[u][v]['weight'] for u, v in G.edges]).reshape(-1, 1), dtype=torch.float)
        node_features = torch.tensor(np.array([G.nodes[i]['features'] for i in G.nodes]), dtype=torch.float)
        u = torch.zeros(dim_global_features, dtype=torch.float).view(-1, 1)
        pyg_data = Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, u=u)
        return pyg_data
    