import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
import os


# Hyperparameters
batch_size = 128
block_size = 64
max_iters = 3000
eval_interval = 600
learning_rate = 1e-4
eval_iters = 500
n_embd = 384
n_layers = 6 # Layer size = n_embd / n_layers = 64
n_heads = 8
dropout = 0.1

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)

torch.manual_seed(42)

# Check for the existence of either txt or csv file
filename = None
if os.path.exists('filtered_messages.txt'):
    filename = 'filtered_messages.txt'
elif os.path.exists('filtered_messages.csv'):
    filename = 'filtered_messages.csv'
else:
    raise FileNotFoundError("Neither 'filtered_messages.txt' nor 'filtered_messages.csv' was found.")
print('Using file:', filename)

# Load data
with open(filename, 'r', encoding='utf-8') as file:
    content = file.read().replace('\n', ' ')

# Create vocabulary
vocab = sorted(set(content))
vocab_size = len(vocab)
print('Vocabulary size:', vocab_size)
print('Vocabulary:', ''.join(vocab))
# Export the vocabulary
with open('vocab.txt', 'w', encoding='utf-8') as file:
    file.write(''.join(vocab))

# Create a dictionary of characters mapped to integers
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}
encoded = lambda s: [stoi[ch] for ch in s] # encode a string
decoded = lambda l: ''.join([itos[i] for i in l]) # decode a list of integers

# Encode the entire content
encoded_content = encoded(content)
encoded_content = torch.tensor(encoded_content, dtype=torch.int64)

# Split the data into training and validation sets
train_size = int(len(encoded_content) * 0.8)
train_data = encoded_content[:train_size]
val_data = encoded_content[train_size:]

# Create a function to generate batches of data
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + 1 + block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# Create a class for the self-attention head
class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) / C**0.5
        wei = wei.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
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

    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads=n_heads) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

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
            targets = targets.view(-1) # B*T
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_tokens=100):
        for _ in range(max_tokens):
            idx_cond = idx[:, -block_size:] # crop the sequence
            logits, loss = self(idx_cond) # predictions
            logits = logits[:, -1, :] # last prediction
            probs = F.softmax(logits, dim=1) # probabilities
            idx_new = torch.multinomial(probs, num_samples=1) # new index
            idx = torch.cat([idx, idx_new], dim=1) # add to the sequence
        return idx