import re

class Tokennizator_v1:

    def preprocessed_text(self, raw_text):
        pre_processed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
        pre_processed = [item for item in pre_processed if item.strip()]
        all_words = sorted(set(pre_processed))
        print(f" Vocabulary size: {len(all_words)}")

        vocab = {
            token: integer for integer, token in enumerate(all_words)
        }
        return vocab

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