import os
import requests
import torch
from torch.utils.data import DataLoader

from .dataset import dataset_v1 as dtt
from .tokenizator import Tokennizator_v1 as tkz
from .attention import self_attention_v1 as sa
from .attention import casual_attention as ca
from .attention import Multi_head_attention as mha

DATA_PATH = "data/the-veredict.txt"
DATA_URL = (
    "https://raw.githubusercontent.com/rasbt/"
    "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
    "the-verdict.txt"
)


def load_text(file_path: str = DATA_PATH, url: str = DATA_URL) -> str:
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        with open(file_path, "wb") as f:
            f.write(response.content)

    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def build_tokenizer(text: str) -> tuple[tkz, dict]:
    vocab = tkz({}).preprocessed_text(text)
    return tkz(vocab), vocab


def data_loader_v1(
    txt: str,
    batch_size: int = 4,
    max_len: int = 256,
    stride: int = 128,
    shuffle: bool = True,
    drop_last: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    tokenizer, _ = build_tokenizer(txt)
    dataset = dtt(txt, tokenizer, max_len, stride)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )


def softmax_naive(x: torch.Tensor) -> torch.Tensor:
    return torch.exp(x) / torch.exp(x).sum(dim=0)


raw_text = load_text()
tokenizer, vocab = build_tokenizer(raw_text)

max_length = 4
dataloader = data_loader_v1(
    raw_text, batch_size=8, max_len=max_length, stride=max_length, shuffle=False
)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("Targets:\n", targets)

# Embedding Layer
vocab_size = len(vocab)
output_dim = 256
context_length = max_length

token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
position_embedding_layer = torch.nn.Embedding(context_length, output_dim)

token_embedding = token_embedding_layer(inputs)
position_embedding = position_embedding_layer(torch.arange(max_length))
inputs_embedding = token_embedding + position_embedding

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

d_in = 3
d_out = 2

sa_example = sa(d_in, d_out)
print(sa_example(inputs))


batch = torch.stack([inputs, inputs], dim= 0)


context_length = batch.shape[1]
ca_example = ca(d_in, d_out,context_length, 0.0)

context_vec = ca_example(batch)
print(context_vec)
print(context_vec.shape)

mha_example = mha(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha_example(batch)

print(context_vecs)

print(context_vecs.shape)