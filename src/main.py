import os
import requests
from .tokenizator import Tokennizator_v1 as tkz
from .dataset import dataset_v1 as dtt
from torch.utils.data import DataLoader


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

dataloader = data_loader_v1 (
    raw_text, batch_size=8, max_len=4, stride=1, shuffle=False
)

data_iter = iter(dataloader)

inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("Targets:\n", targets)