import argparse
import numpy as np
import pandas as pd
import time
import wandb


from sklearn.metrics import average_precision_score
from concrete.ml.common.serialization.loaders import load

from src.transformer_vectorizer import TransformerVectorizer
from src.gnn_vectorizer import GNNVectorizer


def load_data(emb_name: str) -> tuple:
    if emb_name == "transf":
        x_train = np.load("data/preprocessed_data/x_train_transf_emb.npy")
        x_test = np.load("data/preprocessed_data/x_test_transf_emb.npy")
    elif emb_name == "gnn":
        x_train = np.load("data/preprocessed_data/x_train_gnn_emb.npy")
        x_test = np.load("data/preprocessed_data/x_test_gnn_emb.npy")
    elif emb_name == "transf_gnn":
        x_train = np.load("data/preprocessed_data/x_train_transf_gnn_emb.npy")
        x_test = np.load("data/preprocessed_data/x_test_transf_gnn_emb.npy")
    y_test = np.load("data/preprocessed_data/y_test.npy")
    return x_train, x_test, y_test

def load_vectorizers(emb_name: str) -> list:
    if emb_name == "transf":
        vectorizers = [TransformerVectorizer()]
    elif emb_name == "gnn":
        vectorizers = [GNNVectorizer()]
    elif emb_name == "transf_gnn":
        vectorizers = [TransformerVectorizer(), GNNVectorizer()]
    return vectorizers

def load_model(emb_name: str, n_bits: int):
    model_name = f"models/xgb_{emb_name}_{n_bits}_bits"
    x_train, _, _ = load_data(emb_name)
    with open(model_name, "r") as f:
        model = load(f)
    model.compile(x_train)
    return model

def evaluate_models(model, config):    
    # Load the test data
    _, x_test, y_test = load_data(config.emb_name)
    
    # Load generated tweets
    tweets = pd.read_csv("data/generated_tweets.csv")

    # Evaluate the model on the test set
    start_time = time.time()
    y_proba = model.predict_proba(x_test)
    end_time = time.time()
    processing_time = end_time - start_time
    
    y_pred = np.argmax(y_proba, axis=1)
    test_acc = np.mean(y_pred == y_test)

    y_pred_positive = y_proba[:, 2]
    y_pred_negative = y_proba[:, 0]
    y_pred_neutral = y_proba[:, 1]

    ap_positive = average_precision_score((y_test == 2), y_pred_positive)
    ap_negative = average_precision_score((y_test == 0), y_pred_negative)
    ap_neutral = average_precision_score((y_test == 1), y_pred_neutral)

    wandb.log({
        "test_processing_time": processing_time,
        "test_accuracy": test_acc,
        "test_ap_positive": ap_positive,
        "test_ap_negative": ap_negative,
        "test_ap_neutral": ap_neutral
        })

    # Evaluate the model on 5 examples from the test set using FHE
    start_time = time.time()
    y_fhe_proba = model.predict_proba(x_test[:10], fhe="execute")
    end_time = time.time()
    fhe_processing_time = end_time - start_time
    
    y_fhe_pred = np.argmax(y_fhe_proba[:10], axis=1)
    is_equal = np.alltrue(y_pred[:10] == y_fhe_pred)

    wandb.log({
        "10 examples_fhe_processing_time": fhe_processing_time,
        "fhe_pred_is_equal_pred": is_equal
        })


    vectorizers = load_vectorizers(config.emb_name)
    # Evaluate the model on the generated dataset with vectorization and FHE
    for n_tweets in [1, 5, 10, 20]:
        start_time = time.time()
        for tweet in tweets[:n_tweets]:
            # Process the tweet and predict
            tweet_emb = [v.transform([tweet]) for v in vectorizers]
            tweet_emb = np.concatenate(tweet_emb, axis=1)
            _ = model.predict(tweet_emb, fhe="execute")

        end_time = time.time()
        processing_time = end_time - start_time
        # Log the results
        wandb.log({"tweets_processing_time": processing_time, "n_tweets": n_tweets})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model evaluation with WandB.")
    parser.add_argument('--entity', type=str, required=True, help='WandB entity name')
    parser.add_argument('--project', type=str, required=True, help='WandB project name')
    args = parser.parse_args()

    n_bits_values = [2, 4, 6, 8]
    emb_name_values = ["transf", "gnn", "transf_gnn"]
    
    for n_bits in n_bits_values:
        for emb_name in emb_name_values:
            # Load model
            model = load_model(emb_name, n_bits)

            # Initialize WandB
            wandb.init(project=args.project, entity=args.entity, config={"emb_name": emb_name, "n_bits": n_bits})
            
            # Evaluate the model
            evaluate_models(model, wandb.config)

            # Finish WandB run
            wandb.finish()
