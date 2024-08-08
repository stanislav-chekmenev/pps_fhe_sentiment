# Privacy-Preserving Sentiment Analysis with Fully Homomorphic Encryption (FHE)

### Description

This repository is dedicated to showcasing machine learning with FHE. It envestigates performance of different models, using Tweets Airline dataset. 

### Installation

```bash
git clone https://github.com/stanislav-chekmenev/pps_fhe_sentiment
cd pps_fhe_sentiment
```
Perform manual installation of Pytorch and Pytorch-Geometric to avoid conflicts with Concrete-ML library's dependencies.

```bash
pip install torch==1.13.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.1+cpu.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.13.1+cpu.html
pip install git+https://github.com/pyg-team/pytorch_geometric.git
```
Then install the requirements.

```bash
pip install -r requirements.txt
```

### Design choices

The FHE library used is Concrete-ML from Zama.ai. This library has a simple enough functionality, a good compatibility with Sklearn and Pytorch and a well-written documentation. The downside of the library that it can't do training on encrypted data.

The XGB classifier model was enchanced not only with transformer embeddings, but also with additional embeddings that were generated by a Graph Neural Network, based on the general Message Passing Neural Network architecture. This type of a GNN can capture complex dependencies within a graph structure. This example demonstrates that one can use a combined approach to text sentiment analysis, employing not ony standard LLMs, but also utilizing any additional relational data within a collection of texts. For more information on graph construction refer to the provided WandB [report](https://wandb.ai/stanislav-chekmenev/gnn_training/reports/GNN-training--Vmlldzo2MDAzMTA1?accessToken=r6egpn6tjakepuk43laqnz7j0h8rfb7jmfekf04y6txw392mkvij9i44l3uv1uzw). The report describing some points of evaluation of FHE models' performance is available [here](https://api.wandb.ai/links/stanislav-chekmenev/jhdpgevk)

### How to run

Before running anything one needs to download the preprocessed data folder. This folder contains embedded numpy arrays with train and test sets that are needed for training, but most importantly they are used for initialization of the model. Also there are subgraphs of the big graph of connected tweets. Those subgraphs are needed for the GNN vectorizer to predict for a newly created node. The data is located under this [link](https://drive.google.com/drive/folders/1WPjV2_LI7nGycK70L1j2a_qfYoXr3xef?usp=drive_link).

One can run the evaluation, executing `evaluate.py` script. One needs to have set up a WandB account, which is free for private use. The documentation how to do it is given on their [website](https://wandb.ai/site/). One has to run this command to execute the script:

```bash
python evaluate.py --entity <my_entity> --project <my_project>
```
where my_entity is an entity name from WandB and project is a WandB project name that can be anything.

One can also retrain the GNN model and the XGB final models, using the scripts `train_gnn_wandb.py` and `train_xgb.py`. For training a GNN one needs to run the following command:

```bash
python train_gnn_wandb.py --entity <my_entity> --project <my_project>
```


