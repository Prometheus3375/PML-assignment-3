# Installation

1. Install [Python 3.9](https://www.python.org/downloads/).
1. Open terminal in the root directory of this assignment.
1. (Optional) Initialize virtual environment and activate it according to the
   [tutorial](https://docs.python.org/3/library/venv.html).
1. Run `python -m pip install -U pip setuptools wheel` to update pip, setuptools and wheel packages.
1. Install necessary NVIDIA drivers to allow PyTorch use GPU.
   *Note*: you do not need to install CUDA binaries.
    - [GPUs with CUDA](https://developer.nvidia.com/cuda-gpus). *This is not a complete list.*
    - [CUDA executables and documentation](https://developer.nvidia.com/cuda-downloads).
    - Read [docs](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
      to find out which drivers you need to support necessary CUDA version.
1. Install [PyTorch 1.8.x](https://pytorch.org/) with CUDA supported by your graphic card and drivers.

# Train data

1. [Yandex English-Russian corpus](https://translate.yandex.ru/corpus?lang=en)
1. [ParaCrawl Russian Corpus](https://www.paracrawl.eu/)

Unzip downloaded archives and place text files in the following directories:

- Yandex: `datasets/yandex`
- ParaCrawl: `datasets/paracrawl`

# Model

We used basic sequence-to-sequence RNN-based model with an attention layer. Our model is based on several tutorials
([1](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html),
[2](https://github.com/bentrevett/pytorch-seq2seq/blob/master/3%20-%20Neural%20Machine%20Translation%20by%20Jointly%20Learning%20to%20Align%20and%20Translate.ipynb))
, but there are some differences.

Some notable changes:

- **Proper batching**. Sentences must have equal length to be placed a batch, i.e. they must be padded with some special
  token. In embeddings and criterion this token is marked as ignored to reduce the impact of padding on model's
  performance. Also, batches are wrapped in PackedSequence object before running through RNNs, and attention layer uses
  a mask to ignore pad tokens.

- **Different model structure**. Although the idea of the model is the same as in tutorials, significant changes are
  made to its structure. All parts of Seq2Seq model:
  Encoder, Attention and Decoder - are independent of each other. Seq2Seq connects them inside resulting in a very
  simple interface. This simplifies training and testing scripts, making them shorter and cleaner.

## Structure

### Encoder

#### Layers

- Embedding
- Dropout (50%)
- GRU
- Linear (`encoder hidden state -> decoder hidden state`)

#### Input

- Encoded Russian sentences
- Lengths of each Russian sentence without padding

#### Output

- GRU output
- Hidden state of shape appropriate for the decoder

### Attention

#### Layers

- Linear (`encoder output + decoder hidden state -> decoder hidden state`)
- Linear (`decoder hidden state -> 1`)
- Softmax

#### Input

- Encoder's GRU output
- Padding mask
- Hidden state

#### Output

- Attention tensor

### Decoder

#### Layers

- Embedding
- Dropout (50%)
- GRU
- Linear (`decoder hidden state + encoder hidden state + decoder embed size -> vocabulary`)

#### Input

- Encoded tokens
- Encoder's GRU output weighted with attention

#### Output

- Predicted tokens
- Hidden state

### Seq2Seq

#### Layers

- Encoder
- Attention
- Decoder

#### Input

- Encoded Russian sentences
- Lengths of each Russian sentence without padding
- Encoded English sentences
- Teaching threshold (number)

#### Output

- Predicted sentences

# Training and results

## Data preparation

- **Dataset slicing**. We implemented dataset slicing allowing reading and using only a part of a whole dataset. This
  helps when the whole dataset is too large or when it is necessary to perform some quick tests.

- **Preprocessing**. Sentences are cleaned from most non-alphanumeric symbols. Numbers and English shortcuts such as
  "don't" and "he's" have special preprocessing.

- **Word dropping**. Words that was met once are dropped from the vocabulary.

## Optimizer and loss function

Adam's optimizer and CrossEntropyLoss with sum reduction.

## Results

Unfortunately, the performance of the model is very low, 1.18 BLEU. Several probable reasons:

- **Small vocabulary**. We have too small GRU RAM. Therefore, only 10k sentences from Yandex dataset were used in
  training. This is too few sentences, resulting in a very small vocabulary. Even with such vocabulary, allowed batch
  size is 60, and 20 epochs of training require 40-50 minutes on GPU.

- **NIL token**. This token represents all words that are not in the vocabulary. Due to words dropping, this token
  starts to represent many words from datasets. Probably, it is better to drop unknown words when transforming a
  sentence to a tensor.
