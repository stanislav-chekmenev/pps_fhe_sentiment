import numpy as np
import pandas as pd
import torch
import tqdm

from sklearn.model_selection import train_test_split 
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Tuple

from src.gnn_vectorizer import GNNVectorizer


class Preprocessor():

    def __init__(self) -> None:
        tknz_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        self.device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.transformer_model = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment-latest")
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(tknz_name)
        self.gnn_vectorizer = GNNVectorizer()

    def load_data(self, data_dir: str) -> Tuple[np.ndarray, ...]:
        train = pd.read_csv(data_dir, index_col=0)
        text_X = train["text"]
        y = train["airline_sentiment"]
        y = y.replace(["negative", "neutral", "positive"], [0, 1, 2])

        text_X_train, text_X_test, y_train, y_test = train_test_split(
            text_X, y, test_size=0.1, random_state=42
            )
        return text_X_train, text_X_test, y_train, y_test

    def get_transformer_embeddings(
            self,
            text: pd.Series,
            ) -> np.ndarray:
        text_list = text.tolist()
        tokenized_text_split = []
        for single_text in text_list:
            tokenized_text_split.append(self.tokenizer.encode(single_text, return_tensors="pt"))

        transformer_model = self.transformer_model.to(self.device)
        output_hidden_states_list = []

        for tokenized_x in tqdm.tqdm(tokenized_text_split):
            output_hidden_states = transformer_model(tokenized_x.to(self.device), output_hidden_states=True)[1][-1]
            # Average over the tokens axis to get a representation at the text level.
            output_hidden_states = output_hidden_states.mean(dim=1)
            output_hidden_states = output_hidden_states.detach().cpu().numpy()
            output_hidden_states_list.append(output_hidden_states)

        return np.concatenate(output_hidden_states_list, axis=0)
    
    def get_gnn_embeddings(self, text: pd.Series) -> np.ndarray:
        text_list = text.tolist()                
        embeddings = []
        for single_text in tqdm.tqdm(text_list):
            embedding = self.gnn_vectorizer.text_to_tensor([single_text])
            embedding = embedding.detach().cpu().numpy()
            embeddings.append(embedding)
        return np.concatenate(embeddings, axis=0)