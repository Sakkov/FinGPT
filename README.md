# TeekkariGPT
TeekkariGPT is a Python script that uses a super small GPT model to generate nonsensical text imitating intoxicated teekkari (Finnish engineering students) conversations. The model is trained on a dataset of messages extracted from Telegram exports.

## Requirements
- Python 3.7+
- PyTorch
- tqdm

## Installation
1. Install the required libraries:
```bash
pip install -r requirements.txt
```
2. Clone the repository or download the bigram.py, train.py, and generate_text.py files.

## Usage
1. Download and prepare a dataset from Telegram or other sources. You can use the TelegramMessageExtractor (https://github.com/Sakkov/telegram-data) to extract messages from the HTML files.

2. Train the model using the train.py script:

```bash
python train.py
```

3. Generate text using the generate_text.py script:

```bash
python generate_text.py
```

You can alternatively use the provided model to generate text by skipping straight to the text generation step.

## Data Loading

The script supports loading data from either a .txt file or a .csv file. It will first look for a file named 'filtered_messages.txt'. If it doesn't find one, it will try to load a file named 'filtered_messages.csv'. If neither file is found, the script will raise a FileNotFoundError.

To use the script, place your data in a file named 'filtered_messages.txt' or 'filtered_messages.csv' in the same directory as the script.

## Model Architecture
TeekkariGPT uses a simplified GPT model with the following components:

- Token and position embeddings
- Self-attention heads (MultiHead)
- Feed-forward layers
- Layer normalization
- The model architecture and hyperparameters are adjustable in the bigram.py script.
- The model uses characters as tokens.

## Example
Here is an example of generated text using the default settings:

```bash
python generate_text.py
```

```output
😍🤩mazing testa10 it 100 all of cä- wete pactHuhumMyteksin Hissinganinen perus looks gonstartTumppa Yrjotella 👌 muuten aptrii pyhämyyneeitä6.  Kokouksen päättäminenPJ päättää kokouksen klo  21.40"
"Gambina meeeting-ikinä innopullo kehustaaBmFerena., SonjaMakotsu2. Kokouksen laillisuus ja päätösvaltaisuusKokous on laillinen ja päätösvaltainen:-Dommi löytyy jätv-pidentää-kyyksen pyhän kaljansa kokouksessa tekee hemmunen tarrastasissaGambina flowarly unisia is the meeting)elymköin
Vitun Gaua?
Erittäin kans
Jaa tää noinkin nykystä hyinkön. Hervantagambiina alkaa t ovat henkerginKyä pois tykkä ystävien jäsenten hyväksyminen- Petu esillä hande one palvessaTiteltä on saunatustaat hyi vittu ja ilmoituksenssaNegenddrali ei tarjottava gonorma on home losa4.  Uusien kannatusjäsenten hyväksyminen6 kokouksen on hyvät päättämminen ajassa täy baarin agumb2:luumpiaPe ""teep kusisut enroosta gambinaa lasays ensimmää, myös pöykköt pullostapveri on ite löytyy tobea.-iidii ryys ryhmä keskustellaan?Uusien k
```

The hyperparameters used in the example above are:

```python
# Hyperparameters
batch_size = 128
block_size = 64
max_iters = 3000
eval_interval = 600
learning_rate = 1e-4
eval_iters = 500
n_embd = 384
n_layers = 6
n_heads = 8
dropout = 0.1
```

The validation loss is about 1.5 and the training loss about 1.5 after 3000 iterations. At 3000 iterations the model is not overfitting yet. I would recommend training the model for at least 10 000 iterations.

## Contributing
If you would like to contribute to the project or suggest improvements, please submit a pull request or create an issue. Your feedback is always welcome!

## License
This project is licensed under the MIT License.