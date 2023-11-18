import logging
import networkx as nx
import numpy as np
import pickle
import torch
import tqdm

from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SentimentGraphBuilder:
    def __init__(self, transformer_embeddings: np.ndarray, labels: np.ndarray):
        self.transformer_embeddings = transformer_embeddings
        self.labels = np.array(labels)

    def create_sentiment_graph(self, similarity_threshold: float = 0.8) -> list:
        logger.info(f"Creating sentiment graph with similarity threshold: {similarity_threshold}")
        num_embeddings = len(self.transformer_embeddings)
        chunk_size = 128  # size of each subgraph
        subgraphs = []

        # Iterate over chunks to create subgraphs
        for start_idx in range(0, num_embeddings, chunk_size):
            end_idx = min(start_idx + chunk_size, num_embeddings)
            G = nx.Graph()

            # Add nodes to the subgraph
            logger.info(f"Adding nodes to subgraph {start_idx // chunk_size + 1}")
            for i in tqdm.tqdm(range(start_idx, end_idx)):
                G.add_node(i - start_idx, features=self.transformer_embeddings[i], labels=self.labels[i])

            logger.info(f"Adding edges to subgraph {start_idx // chunk_size + 1}")
            # Add edges based on cosine similarity
            for i in range(start_idx, end_idx):
                for j in range(i + 1, end_idx):
                    similarity = cosine_similarity(
                        self.transformer_embeddings[i].reshape(1, -1), 
                        self.transformer_embeddings[j].reshape(1, -1)
                    )[0][0]

                    if similarity >= similarity_threshold:
                        # Adjust indices for the subgraph
                        new_i = i - start_idx
                        new_j = j - start_idx
                        G.add_edge(new_i, new_j, weight=similarity)
            
            subgraphs.append(G)
            logger.info(f"Subgraph {start_idx // chunk_size + 1} created successfully")

        logger.info("All subgraphs created successfully")
        return subgraphs
    
    def to_pyg_data(self, G: nx.Graph, dim_global_features: int) -> Data:
        # Extract edge indices and attributes
        edge_index = torch.tensor(list(G.edges)).t().contiguous()
        edge_attr = torch.tensor(np.array([G[u][v]['weight'] for u, v in G.edges]).reshape(-1, 1), dtype=torch.float)

        # Assuming node features are already set in the graph
        node_features = torch.tensor(np.array([G.nodes[i]['features'] for i in G.nodes]), dtype=torch.float)
        node_labels = torch.tensor(np.array([G.nodes[i]['labels'] for i in G.nodes]), dtype=torch.long)

        # Global attribute 'u'
        u = torch.zeros(dim_global_features, dtype=torch.float).view(-1, 1)

        # Create PyTorch Geometric Data
        pyg_data = Data(x=node_features, y=node_labels, edge_index=edge_index, edge_attr=edge_attr, u=u)
        return pyg_data
    
    @staticmethod
    def create_pyg_dataloaders(data_list, batch_size=8, train_split=0.8, shuffle=True):
        # Shuffle the data_list if shuffle is True
        if shuffle:
            np.random.shuffle(data_list)

        # Determine split sizes for training and validation sets
        train_size = int(len(data_list) * train_split)

        # Split the data_list into training and validation sets
        train_data_list = data_list[:train_size]
        val_data_list = data_list[train_size:]

        # Create dataloaders for training and validation sets
        train_dataloader = DataLoader(train_data_list, batch_size=batch_size, shuffle=shuffle)
        val_dataloader = DataLoader(val_data_list, batch_size=batch_size, shuffle=False)

        return train_dataloader, val_dataloader
    
    @staticmethod
    def load(path):
        with open(path, 'rb') as file:
            G = pickle.load(file)
        return G
    

if __name__ == "__main__":
    transformer_embeddings = np.load("data/preprocessed_data/x_train.npy")
    labels = np.load("data/preprocessed_data/y_train.npy")
    graph_builder = SentimentGraphBuilder(transformer_embeddings[:4000], labels[:4000])
    G = graph_builder.create_sentiment_graph()
    with open("data/preprocessed_data/train_subgraphs.pkl", "wb") as f:
        pickle.dump(G, f)
