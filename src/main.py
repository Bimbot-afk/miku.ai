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

inputs_example = torch.tensor(
  [[0.43, 0.15, 0.89],
   [0.55, 0.87, 0.66], 
   [0.57, 0.85, 0.64], 
   [0.22, 0.58, 0.33], 
   [0.77, 0.25, 0.10], 
   [0.05, 0.80, 0.55]] 
)


query = inputs_example[1]

attn_scores_2 = torch.empty(inputs_example.shape[0])
for i, x_i in enumerate(inputs_example):
    attn_scores_2[i] = torch.dot(query, x_i)

print(attn_scores_2)

res = 0

for idx, element in enumerate(inputs_example[0]):
    res += inputs_example[0][idx] * query[idx]

print(res)
print(torch.dot(inputs_example[0], query))

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)

print("Attention Weights 2 Naive:\n", attn_weights_2_naive)
print("sum:\n", attn_weights_2_naive.sum())

context_vec_2 = torch.zeros(query.shape[0])

for i, x_i in enumerate(inputs_example):
    context_vec_2 += attn_weights_2_naive[i] * x_i

print("Context Vector 2:\n", context_vec_2)


attn_scores = inputs_example @ inputs_example.T
attn_weights = torch.softmax(attn_scores, dim=1)
all_context_vecs = attn_weights @ inputs_example

print("all_context_vecs:\n", all_context_vecs)

x_2 = inputs_example[1]
d_in = inputs_example.shape[1]
d_out = 2

torch.manual_seed(123)

w_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = True)
w_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = True)
w_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = True)

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

d_k = keys.shape[1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=1)
print(attn_weights_2)

