import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from bigram import BigramLanguageModel, n_embd, n_heads, n_layers, block_size, device, max_iters, eval_interval, eval_iters, learning_rate, dropout, fine_tune_iters, fine_tune_lr, preprocessDataPath, fineTuneDataPath

# Create a function to estimate loss
@torch.no_grad()
def loss_estimate():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            X, Y = model.get_batch(split)
            _, loss = model(X, Y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = BigramLanguageModel([preprocessDataPath, fineTuneDataPath])
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
    
    xb, yb = m.get_batch('train')
    
    # Evaluate the model
    logits, loss = m(xb, yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

# Save the model
torch.save(m.state_dict(), 'model_preprocessed.pt')

# Loss estimate
losses = loss_estimate()
print("Final loss estimate (preprocessed):")
print(f"Step {steps}: train loss = {losses['train']:.4f}, val loss = {losses['val']:.4f}")

# Fine-tune the model
print("Fine-tuning...")
# Set the learning rate
for param_group in opt.param_groups:
    param_group['lr'] = fine_tune_lr
m.preprocess(source=1) # 1 = fine-tune
m.train()
for steps in tqdm(range(fine_tune_iters), desc="Fine-tuning", unit="step"):

    # Estimate loss periodically
    if steps % eval_interval == 0:
        losses = loss_estimate()
        print(f"Step {steps}: train loss = {losses['train']:.4f}, val loss = {losses['val']:.4f}")

    xb, yb = m.get_batch('train')
    
    # Evaluate the model
    logits, loss = m(xb, yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()

# Loss estimate
losses = loss_estimate()
print("Final loss estimate (fine-tuned):")
print(f"Step {steps}: train loss = {losses['train']:.4f}, val loss = {losses['val']:.4f}")

# Generate text
context = torch.zeros((1,1), dtype=torch.int64, device=device)
output_encoded = m.generate(context, max_tokens=1000)[0].tolist()
output = m.decoded(output_encoded)
print(output)

# Save the model
torch.save(m.state_dict(), 'model_fine_tuned.pt')