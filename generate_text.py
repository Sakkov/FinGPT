import torch
from bigram import BigramLanguageModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model = BigramLanguageModel().to(device)
model.load_state_dict(torch.load('model.pt', map_location=device))
model.eval()

# Load vocabulary
with open('vocab.txt', 'r') as f:
    vocab = f.read()

# Create vocabulary
vocab = sorted(set(vocab))
vocab_size = len(vocab)
# Create a dictionary of characters mapped to integers
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}
encoded = lambda s: [stoi[ch] for ch in s] # encode a string
decoded = lambda l: ''.join([itos[i] for i in l]) # decode a list of integers


# Generate text
context = torch.zeros((1,1), dtype=torch.int64, device=device)
print(decoded(model.generate(context, max_tokens=1000)[0].tolist()))