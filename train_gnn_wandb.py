import argparse
import logging
import numpy as np
import os
import torch
import tqdm
import wandb
import yaml

from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from src.graph.build_graph import SentimentGraphBuilder
from src.graph.mpnn import MPNN
from src.graph.gat import GATModel


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_gnn(transformer_embeddings, labels, config) -> tuple:
    # Use WandB configuration
    learning_rate = config["learning_rate"]
    model_type = config["model_type"]
    num_passes = config["num_passes"]

    # Specify other vars
    EPOCHS = 28
    INPUT_NODE_DIM = int(transformer_embeddings.shape[1])
    INPUT_EDGE_DIM = 1
    INPUT_GLOBAL_DIM = 1
    DIM_NODE_FEATURES = 128
    DIM_EDGE_FEATURES = 32
    DIM_GLOBAL_FEATURES = 32
    HIDDEN_CHANNELS = 32
    NUM_CLASSES = len(set(labels))
    NUM_HEADS = 8

    # Load a prebuilt a graph
    graph_builder = SentimentGraphBuilder(transformer_embeddings, labels)
    subgraphs = graph_builder.load(os.path.join("data","preprocessed_data","train_subgraphs.pkl"))
    pyg_data_list = [graph_builder.to_pyg_data(subgraph, 1) for subgraph in subgraphs]
    train_loader, val_loader = graph_builder.create_pyg_dataloaders(pyg_data_list)

    if model_type == 'MPNN':
        # Train MPNN model
        num_passes = config["num_passes"]
        mpnn_model = MPNN(
            INPUT_NODE_DIM, INPUT_EDGE_DIM, INPUT_GLOBAL_DIM,
            DIM_NODE_FEATURES, DIM_EDGE_FEATURES, DIM_GLOBAL_FEATURES,
            HIDDEN_CHANNELS, NUM_CLASSES, num_passes)
        model = mpnn_model
    elif model_type == 'GAT':
        # Train GAT model
        gat_model = GATModel(INPUT_NODE_DIM, HIDDEN_CHANNELS, NUM_CLASSES, NUM_HEADS)
        model = gat_model
    else:
        raise ValueError("Invalid model type. Choose 'MPNN' or 'GAT'.")

    # Training loop
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), learning_rate)

    logger.info(f"Starting training.")
   # Training loop
    for epoch in tqdm.tqdm(range(EPOCHS)):
        model.train()
        total_loss = 0
        total_correct = 0
        total_examples = 0

        for pyg_batch in train_loader:
            optimizer.zero_grad()
            out, _ = model(pyg_batch)
            loss = criterion(out, pyg_batch.y)
            pred = out.argmax(dim=1)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_correct += int((pred == pyg_batch.y).sum())
            total_examples += pyg_batch.y.size(0)

        # Calculate training accuracy and average loss
        train_acc = total_correct / total_examples
        avg_train_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_examples = 0
        with torch.no_grad():
            for pyg_batch in val_loader:
                val_out, _ = model(pyg_batch)
                val_loss += criterion(val_out, pyg_batch.y).item()
                pred = val_out.argmax(dim=1)
                val_correct += int((pred == pyg_batch.y).sum())
                val_examples += pyg_batch.y.size(0)

        val_acc = val_correct / val_examples
        avg_val_loss = val_loss / len(val_loader)

        # Log metrics
        wandb.log({
            "train_loss": avg_train_loss,
            "train_acc": train_acc, 
            "val_loss": avg_val_loss,
            "val_acc": val_acc,
            "epoch": epoch + 1}
        )

    logger.info(f"Training is completed.")
    return model

def train():
    # Initialize WandB
    wandb.init(reinit=True)

    # Setup WandB sweep config 
    config = wandb.config

    # Load data
    transformer_embeddings = np.load(os.path.join("data","preprocessed_data","x_train_transf_emb.npy"))
    labels = np.load(os.path.join("data","preprocessed_data","y_train.npy"))
    
    # Train and tune GNNs
    model = train_gnn(transformer_embeddings, labels, config)
    return model

# Fetch the best run configuration using WandB API
def retrain_with_best_config(wandb_entity, wandb_project, sweep_id):
    wandb.init(reinit=True)

    transformer_embeddings = np.load(os.path.join("data","preprocessed_data","x_train_transf_emb.npy"))
    labels = np.load(os.path.join("data","preprocessed_data","y_train.npy"))

    api = wandb.Api()
    sweep = api.sweep(f"{wandb_entity}/{wandb_project}/{sweep_id}")
    best_run = sorted(sweep.runs, key=lambda r: r.summary.get("val_acc", 0), reverse=True)[0]
    best_model = train_gnn(transformer_embeddings, labels, best_run.config)
    torch.save(best_model.state_dict(), 'models/best_gnn.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GNN model training with WandB.")
    parser.add_argument('--entity', type=str, required=True, help='WandB entity name')
    parser.add_argument('--project', type=str, required=True, help='WandB project name')
    args = parser.parse_args()

    with open(os.path.join('sweep_configs', 'sweep_config_gnn.yaml'), 'r') as file:
        sweep_config = yaml.safe_load(file)

    # Initialize a sweep
    sweep_id = wandb.sweep(sweep=sweep_config, project=args.project, entity=args.entity)
    # Run the sweep
    wandb.agent(sweep_id, train, count=40)

    # Retrain with the best params
    retrain_with_best_config(args.entity, args.project, sweep_id)