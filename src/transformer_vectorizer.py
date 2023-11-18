import numpy
import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer

class TransformerVectorizer:
    def __init__(self):
        # Load the tokenizer (converts text to tokens)
        self.tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

        # Load the pre-trained model
        self.transformer_model = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def text_to_tensor(self, texts: list) -> numpy.ndarray:
        # First, tokenize all the input text
        tokenized_text_X_train = self.tokenizer.batch_encode_plus(texts, return_tensors="pt")["input_ids"]

        tokenized_text_X_train_split = torch.split(tokenized_text_X_train, split_size_or_sections=50)

        # Send the model to the device
        transformer_model = self.transformer_model.to(self.device)
        output_hidden_states_list = []

        for tokenized_x in tokenized_text_X_train_split:
            output_hidden_states = transformer_model(tokenized_x.to(self.device), output_hidden_states=True)[1][-1]
            # Average over the tokens axis to get a representation at the text level.
            output_hidden_states = output_hidden_states.mean(dim=1)
            output_hidden_states = output_hidden_states.detach().cpu().numpy()
            output_hidden_states_list.append(output_hidden_states)

        self.encodings = numpy.concatenate(output_hidden_states_list, axis=0)
        return self.encodings

    def transform(self, texts: list):
        return self.text_to_tensor(texts)

