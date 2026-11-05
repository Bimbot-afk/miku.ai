import os
import requests
import torch
from torch.utils.data import DataLoader

from .dataset import dataset_v1 as dtt
from .tokenizator import Tokennizator_v1 as tkz

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

inputs_example = torch.tensor(
    [
        [0.43, 0.15, 0.89],
        [0.55, 0.87, 0.66],
        [0.57, 0.85, 0.64],
        [0.22, 0.58, 0.33],
        [0.77, 0.25, 0.10],
        [0.05, 0.80, 0.55],
    ]
)

query = inputs_example[1]

attn_scores_2 = torch.empty(inputs_example.shape[0])
for i, x_i in enumerate(inputs_example):
    attn_scores_2[i] = torch.dot(query, x_i)

print(attn_scores_2)

res = sum(
    inputs_example[0][idx] * query[idx] for idx in range(len(inputs_example[0]))
)

print(res)
print(torch.dot(inputs_example[0], query))

attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention Weights 2 Naive:\n", attn_weights_2_naive)
print("sum:\n", attn_weights_2_naive.sum())

context_vec_2 = torch.zeros(query.shape[0])
for i, x_i in enumerate(inputs_example):
    context_vec_2 += attn_weights_2_naive[i] * x_i

print("Context Vector 2:\n", context_vec_2)

attn_scores = inputs_example @ inputs_example.T
attn_weights = torch.softmax(attn_scores, dim=-1)
all_context_vecs = attn_weights @ inputs_example

print("all_context_vecs:\n", all_context_vecs)

x_2 = inputs_example[1]
d_in = inputs_example.shape[1]
d_out = 2

torch.manual_seed(123)

w_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)
w_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)
w_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=True)

query_2 = x_2 @ w_query
key_2 = x_2 @ w_key
value_2 = x_2 @ w_value

print(query_2)

keys = inputs_example @ w_key
values = inputs_example @ w_value

print("Keys shape:", keys.shape)
print("Values shape:", values.shape)

key_2 = keys[1]
attn_score_22 = query_2.dot(key_2)
print(attn_score_22)

attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)

d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print(attn_weights_2)
