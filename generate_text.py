import torch
from bigram import BigramLanguageModel
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

preprocessDataPath = 'filtered_training_data/Finnish/'
fineTuneDataPath = 'filtered_training_data/Finnish/filtered_messages_2023-07-24_aivansama.txt'
modelFile = 'model_preprocessed.pt'

# Load the model
model = BigramLanguageModel([preprocessDataPath, fineTuneDataPath]).to(device)
model.load_state_dict(torch.load(modelFile, map_location=device))
model.eval()

# Generate text
context = torch.zeros((1,1), dtype=torch.int64, device=device)
print(model.decoded(model.generate(context, max_tokens=1000)[0].tolist()))