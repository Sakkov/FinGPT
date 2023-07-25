import torch
from bigram import BigramLanguageModel
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

preprocessDataPath = 'training_data/Finnish/'
fineTuneDataPath = 'training_data/Finnish/filtered_messages_2023-07-24_aivansama.txt'
modelFile = 'model_fine_tuned.pt'

# Load the model
model = BigramLanguageModel([preprocessDataPath, fineTuneDataPath]).to(device)
model.load_state_dict(torch.load(modelFile, map_location=device))
model.eval()

# Generate text
context = torch.zeros((1,1), dtype=torch.int64, device=device)
print(model.decoded(model.generate(context, max_tokens=1000)[0].tolist()))