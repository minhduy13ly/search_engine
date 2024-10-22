from transformers import AutoTokenizer, AutoModel
import torch
from datasets import Dataset
import pandas as pd
import numpy as np
import os
import faiss


# Set environment variable to allow duplicate OpenMP runtime
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

model = AutoModel.from_pretrained('vinai/phobert-base')
tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

def get_embedding(item):
    # Hàm thực hiện map
    # Nhiệm vụ:
    # tokenizer mẫu hiện tại

    # lấy vector embedidng của mẫu
    # Tokenize the sentence
    inputs = tokenizer(item, max_length = 128,
                       truncation = True, padding = 'max_length',
                       return_tensors="pt")

    # Get the outputs from PhoBERT
    with torch.no_grad():
        outputs = model(**inputs)


    # Get the last hidden state (embeddings) from the model
    last_hidden_states = outputs.last_hidden_state  # Shape: [batch_size, sequence_length, hidden_size]

    # Remove padding tokens by using the attention mask
    attention_mask = inputs['attention_mask']  # Shape: [batch_size, sequence_length]
    masked_hidden_states = last_hidden_states * attention_mask.unsqueeze(-1)  # Shape: [batch_size, sequence_length, hidden_size]

    # Compute the mean of token embeddings (ignoring padding tokens)
    mean_embedding = masked_hidden_states.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)

    # Remove the batch_size dimension by squeezing it out (if there's only one sentence)
    mean_embedding = mean_embedding.squeeze(0)

    return mean_embedding


class ReRanker():
    def __init__(self):
        pass

    def rank(self, query, docs):

        # Load các docs từ TF-IDF và chuyển thành Hugging Dataset
        # Thực hiện tìm vector tương đồng với query
        # Trả về kết quả
        df = pd.DataFrame(docs)
        df['vector'] = df['text'].apply(lambda x: get_embedding(x))
        vector_data = np.vstack([tensor.detach().cpu().numpy() for tensor in df['vector']]).astype('float32')

        # Initialize FAISS index (using L2 distance for 128-dimensional vectors)
        dimension = 768
        index = faiss.IndexFlatL2(dimension)  # Using L2 distance
        # Add the vectors to the index
        index.add(vector_data)

        query_vector = get_embedding(query).detach().cpu().numpy().astype('float32').reshape(1, 768)

        distances, indices = index.search(query_vector, 1)

        nearest_rows = df.iloc[indices[0]]

        score = nearest_rows['score'].values[0]

        doc_id = nearest_rows['doc_id'].values[0]

        text = nearest_rows['text'].values[0]

        return doc_id, score, text