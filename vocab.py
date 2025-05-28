import re
from collections import Counter

class Vocabulary:
    def __init__(self, freq_threshold=5):
        self.freq_threshold = freq_threshold
        self.itos = {0:"<pad>",1:"<sos>",2:"<eos>",3:"<unk>"}
        self.stoi = {v:k for k,v in self.itos.items()}
        self.next_index = 4

    def __len__(self):
        return len(self.itos)

    def tokenize(self, text):
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def build_vocabulary(self, sentence_list):
        freq = Counter()
        for sentence in sentence_list:
            freq.update(self.tokenize(sentence))
        for token, cnt in freq.items():
            if cnt >= self.freq_threshold and token not in self.stoi:
                self.stoi[token] = self.next_index
                self.itos[self.next_index] = token
                self.next_index += 1

    def numericalize(self, text):
        tokens = ["<sos>"] + self.tokenize(text) + ["<eos>"]
        return [self.stoi.get(tok, self.stoi["<unk>"]) for tok in tokens]
