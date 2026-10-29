import os
import requests
from .tokenizator import Tokennizator_v1 as tkz
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

enc = tokenizer.encode(raw_text)
print(len(enc))

context_size = 4

enc_sample = enc[:50]

x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f" {x}")
print(f"     {y}")
print("--------------------------------")
print(f"{tokenizer.decode(x)}")
print(f"  {tokenizer.decode(y)}")