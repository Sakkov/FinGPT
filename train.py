import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from bigram import BigramLanguageModel, vocab_size, n_embd, n_heads, n_layers, block_size, decoded, device, max_iters, eval_interval, get_batch, eval_iters, learning_rate



# Create a function to estimate loss
@torch.no_grad()
def loss_estimate():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = BigramLanguageModel()
m = model.to(device)

# PyTorch optimizer
opt = torch.optim.Adam(m.parameters(), lr=learning_rate)

# Train the model
print("Training...")
for steps in tqdm(range(max_iters), desc="Training", unit="step"):
    
    # Estimate loss periodically
    if steps % eval_interval == 0:
        losses = loss_estimate()
        print(f"Step {steps}: train loss = {losses['train']:.4f}, val loss = {losses['val']:.4f}")
    
    xb, yb = get_batch('train') # get a batch of data
    
    # Evaluate the model
    logits, loss = m(xb, yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

# Loss estimate
losses = loss_estimate()
print(f"Step {steps}: train loss = {losses['train']:.4f}, val loss = {losses['val']:.4f}")

# Generate text
context = torch.zeros((1,1), dtype=torch.int64, device=device)
print(decoded(m.generate(context, max_tokens=1000)[0].tolist()))

# Save the model
torch.save(m.state_dict(), 'model.pt')