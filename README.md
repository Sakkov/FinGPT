# FinGPT
FinGPT is a Python script that trains, evaluates, and generates Finnish text. The example model was trained on a dataset of mainly Finnish messages from a Telegram group chat. The model is based on the GPT architecture and uses characters as token.

If you want to play with the produced model FinGPT-1.0, you do so for free at https://fingpt.fi. The generation is happening on super weak metal. If you want to help me out, you can donate some GPU time at https://bmc.link/sakkov.

## Requirements
- Python 3.7+
- PyTorch
- tqdm

## Installation
1. Clone the repository or download the bigram.py, train.py, and generate_text.py files.

2. Install the required libraries:
```bash
pip install -r requirements.txt
```

## Usage
1. Download and prepare a dataset from Telegram or other sources. You can use the TelegramMessageExtractor (https://github.com/Sakkov/telegram-data) to extract messages from the HTML files.

    1.5 (Optional) Filter the dataset using the filter_dataset.py script. The script removes tokens(characters) that are not common in regular Finnish language such as emojis. 

2. Train the model using the train.py script:

```bash
python train.py
```

3. Generate text using the generate_text.py script:

```bash
python generate_text.py
```

You can alternatively use any other compatible model to generate text by skipping straight to the text generation step.

## Data Loading

The script supports loading data from a .txt file. 

In the file bigram.py you can specify the path to the dataset and adjust the following parameters:

```python
preprocessDataPath = 'filtered_training_data/Finnish/wikipedia-fi-2017/'
fineTuneDataPath = 'filtered_training_data/Finnish/filtered_messages.txt'

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
```

## Model Architecture
FinGPT uses a simplified GPT model with the following components:

- Token and position embeddings
- Self-attention heads (MultiHead)
- Feed-forward layers
- Layer normalization
- The model architecture and hyperparameters are adjustable in the bigram.py script.
- The model uses characters as tokens.

## Example
Here is an example of generated text using the example model:

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