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
Here is an example of generated text using the default settings:

```bash
python bigram.py
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