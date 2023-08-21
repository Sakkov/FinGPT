import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import os
import json
import sys

# Default hyperparameters
batch_size = 128
block_size = 128
max_iters = 100000
eval_interval = 10000
learning_rate = 1e-4
eval_iters = 500
n_embd = 512
n_layers = int(n_embd / 64)
n_heads = 8
dropout = 0.2
fine_tune_iters = 10000
fine_tune_lr = 1e-5
train_split = 0.8

# Hyperparameters
batch_size = 128  # Number of sequences processed in parallel
block_size = 256  # Number of tokens in a sequence
max_iters = 5000  # Number of training iterations
eval_interval = 1000  # Number of iterations between evaluations
learning_rate = 1e-3
eval_iters = 500  # Number of iterations to estimate loss
n_embd = 512  # Size of the embeddings
n_layers = int(n_embd / 64)
n_heads = 8  # Number of attention heads
dropout = 0.1  # Dropout rate
fine_tune_iters = 2000  # Number of iterations to fine-tune the model
fine_tune_lr = 1e-5  # Learning rate for fine-tuning
train_split = 0.8  # How much of the data is used for training

# Export hyperparameters as json
hyperparameters = {
    "batch_size": batch_size,
    "block_size": block_size,
    "max_iters": max_iters,
    "eval_interval": eval_interval,
    "learning_rate": learning_rate,
    "eval_iters": eval_iters,
    "n_embd": n_embd,
    "n_layers": n_layers,
    "n_heads": n_heads,
    "dropout": dropout,
    "fine_tune_iters": fine_tune_iters,
    "fine_tune_lr": fine_tune_lr,
    "train_split": train_split,
}
with open("hyperparameters.json", "w") as file:
    json.dump(hyperparameters, file)


preprocessDataPath = "filtered_training_data/Finnish/custom_finGPT-1.2_21_8_2023"
fineTuneDataPath = "filtered_training_data/Finnish/suomi24-2001-2017/s24_2017.vrt.txt"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

torch.manual_seed(42)


# Create a class for the self-attention head
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) / C**0.5
        wei = wei.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


# Create a class for the multi-head attention
class MultiHead(nn.Module):
    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


# Create a class for the feed-forward
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        self.sa_head = MultiHead(n_heads, n_embd // n_heads)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa_head(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


# Simple bigram model
class BigramLanguageModel(nn.Module):
    """
    A simple bigram language model
    """

    def __init__(self, sources):
        """
        Initialize the model

        source: path to the source file
        """
        super().__init__()

        # Read the data
        self.content = ["", ""]
        count = 0
        try:
            for source in sources:
                self.content[count] += self.read_file_to_string(source)
                count += 1
        except FileNotFoundError:
            raise FileNotFoundError(f"File {source} not found")
        except Exception as e:
            raise e

        # Create vocabulary
        self.create_vocab(sources)

        # Create a dictionary of characters mapped to integers
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.itos = {i: ch for i, ch in enumerate(self.vocab)}
        self.encode = lambda s: [self.stoi[ch] for ch in s]  # encode a string
        self.decode = lambda l: "".join(
            [self.itos[i] for i in l]
        )  # decode a list of integers

        # Encode the content
        for source_n in range(len(self.content)):
            self.content[source_n] = self.encode(self.content[source_n])
            self.content[source_n] = torch.tensor(self.content[source_n], dtype=torch.int64)
            print(f"Encoded {len(self.content[source_n])} tokens")

        # Preprocess the data
        self.preprocess(source=0)  # 0 = pretrain, 1 = finetune

        # Create the model
        self.token_embedding = nn.Embedding(self.vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_heads=n_heads) for _ in range(n_layers)]
        )
        self.ln_final = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, self.vocab_size)

    def preprocess(self, source):
        """
        Preprocess the data
        """

        # Split the data into training and validation sets
        self.train_size = int(len(self.content[source]) * train_split)
        print(f"Training set size: {self.train_size}")
        self.train_data = self.content[source][: self.train_size]
        self.val_data = self.content[source][self.train_size :]
        print(f"Validation set size: {len(self.val_data)}")

    def read_file_to_string(self, source):
        """
        Read the file recursively and return the string
        """
        if os.path.isfile(source):
            with open(source, "r", encoding="utf-8", errors="ignore") as file:
                content = file.read()
                print(
                    f"Reading {source} of size {round(os.path.getsize(source)/1048576)}MiB"
                )
                print("Done")
                print(f"Read {len(content)} characters.\n\n")
                return content
        else:
            return "".join(
                self.read_file_to_string(os.path.join(source, filename))
                for filename in os.listdir(source)
            )

    def create_vocab(self, sources):
        """
        Create the vocabulary and dictionaries
        """
        vocab_set = set()
        for content in self.content:
            vocab_set.update(set(content))

        # Create vocabulary
        self.vocab = sorted(vocab_set)
        self.vocab_size = len(self.vocab)
        print("Vocabulary size:", self.vocab_size)
        print("Vocabulary:", "".join(self.vocab))

        # Export the vocabulary
        with open("vocab.txt", "w", encoding="utf-8") as file:
            file.write("".join(self.vocab))

    # Create a function to generate batches of data
    def get_batch(self, split):
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i : i + block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding(idx)
        pos_emb = self.pos_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(-1)  # B*T
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_tokens=100):
        for _ in range(max_tokens):
            idx_cond = idx[:, -block_size:]  # crop the sequence
            logits, loss = self(idx_cond)  # predictions
            logits = logits[:, -1, :]  # last prediction
            probs = F.softmax(logits, dim=1)  # probabilities
            idx_new = torch.multinomial(probs, num_samples=1)  # new index
            idx = torch.cat([idx, idx_new], dim=1)  # add to the sequence
        return idx
