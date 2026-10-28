import os
import requests
import re

if not os.path.exists("the-veredict.txt"):
    url =  ("https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt"
    )
    file_path = "the-veredict.txt"

    response = requests.get(url, timeout=30)
    response.raise_for_status()
    with open(file_path, "wb") as f:
        f.write(response.content)

with open("the-veredict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()


pre_processed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
pre_processed = [item for item in pre_processed if item.strip()]
print(f"Length of pre_processed: {len(pre_processed)} tokens")

all_words = sorted(set(pre_processed))
vocab_size = (len(all_words))


vocab = {
    token: integer for integer, token in enumerate(all_words)
}


class Tokennizator_v1:
    def __init__(self, vocab):
        self.srt_to_int = vocab
        self.int_to_srt = {i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [
            item if item in self.srt_to_int
            else '<unk>'
            for item in preprocessed
        ]
        
        ids = [self.srt_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_srt[i] for i in ids])
        text =  re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
