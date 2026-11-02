import os
import requests
from .tokenizator import Tokennizator_v1 as tkz
from .dataset import dataset_v1 as dtt
from torch.utils.data import DataLoader
import torch


if not os.path.exists("data/the-veredict.txt"):
    url =  ("https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt"
    )
    file_path = "data/the-veredict.txt"

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(file_path, "wb") as f:
        f.write(response.content)

with open("./data/the-veredict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

tokenizer_h = tkz({})
vocab = tokenizer_h.preprocessed_text(raw_text)

tokenizer = tkz(vocab)

def data_loader_v1(txt, batch_size=4, max_len = 256, stride=128, shuffle = True, drop_last = True, num_workers = 0):
    tokenizer_h = tkz({})
    vocab = tokenizer_h.preprocessed_text(raw_text)

    tokenizer = tkz(vocab)

    dataset = dtt(txt, tokenizer, max_len, stride)
    
    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        drop_last = drop_last,
        num_workers = num_workers
    )

    return dataloader


max_length = 4

dataloader = data_loader_v1 (
    raw_text, batch_size=8, max_len=max_length, stride=max_length, shuffle=False
)

data_iter = iter(dataloader)

inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("Targets:\n", targets)

# Embedding Layer
vocab_size = len(vocab)
output_dim = 256


token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
token_embedding = token_embedding_layer(inputs)

context_length = max_length
position_embedding_layer = torch.nn.Embedding(context_length, output_dim)

position_embedding = position_embedding_layer(torch.arange(max_length))

inputs_embedding = token_embedding + position_embedding
print("Inputs Embedding:", inputs_embedding.shape)


your_journey = "your journey starts with one step"
encoded_journey = tokenizer.encode(your_journey)
inpu_journey = torch.tensor(encoded_journey)
print("Input of tokens:\n", inpu_journey)