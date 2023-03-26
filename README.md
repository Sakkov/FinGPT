# TeekkariGPT
TeekkariGPT is a Python script that uses a super small GPT model to generate nonsensical text imitating intoxicated teekkari (Finnish engineering students) conversations. The model is trained on a dataset of messages extracted from Telegram exports.

## Requirements
- Python 3.7+
- PyTorch
- tqdm

## Installation
1. Install the required libraries:
```bash
pip install torch tqdm
```
2. Clone the repository or download the bigram.py file.

## Usage
1. Download and prepare a dataset from Telegram or other sources.

2. Run the bigram.py script:

```bash
python bigram.py
```

You can use any data set as training. I would recommend using a dataset of 1 to 10 million characters. By default the script expects a dataset in the form of a CSV file with a single column. The CSV file should be named "filtered_messages.csv" and placed in the same directory as the script. 

You can also use other file types but you will need to modify the script accordingly.

I used a dataset of about 5 million characters extracted from Telegram HTML exports. I used my other project TelegramMessageExtractor (https://github.com/Sakkov/telegram-data) to extract the messages from the HTML files.

## Model Architecture
TeekkariGPT uses a simplified GPT model with the following components:

- Token and position embeddings
- Self-attention heads (MultiHead)
- Feed-forward layers
- Layer normalization
- The model architecture and hyperparameters are adjustable in the bigram.py script.
- The model uses characters as tokens.

## Example
To be added...

Generated text from TeekkariGPT goes here...

## Contributing
If you would like to contribute to the project or suggest improvements, please submit a pull request or create an issue. Your feedback is always welcome!

## License
This project is licensed under the MIT License.