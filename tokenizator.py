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

print(f"Total number of characters: {len(raw_text)}")

pre_processed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
pre_processed = [item for item in pre_processed if item.strip()]
print(f"Length of pre_processed: {len(pre_processed)} tokens")

all_words = sorted(set(pre_processed))
vocab_size = (len(all_words))

print(vocab_size)

vocab = {
    token: integer for integer, token in enumerate(all_words)
}

for i, item in enumerate(vocab.items()):
    print(item)

