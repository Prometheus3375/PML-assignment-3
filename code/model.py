from collections import OrderedDict

import torch
from torch import Tensor, nn
from torch.nn import functional

from data import Token_PAD
from device import Device


class Model(nn.Module):
    @property
    def data(self, /):
        return self.__getnewargs__(), self.state_dict()

    @classmethod
    def from_data(cls, data: tuple[tuple, OrderedDict[str, Tensor]], /):
        args, weights = data
        model = cls(*args)
        model.load_state_dict(weights)
        return model


class RNN(Model):
    def __init__(self, input_dim: int, hidden_dim: int, layers_count: int, bi: bool, /):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=layers_count,
            bidirectional=bi,
        )

    def __getnewargs__(self, /):
        return self.input_dim, self.hidden_dim, self.layers_count, self.bi

    @property
    def input_dim(self, /):
        return self.lstm.input_size

    @property
    def hidden_dim(self, /):
        return self.lstm.hidden_size

    @property
    def layers_count(self, /):
        return self.lstm.num_layers

    @property
    def bi(self, /):
        return self.lstm.bidirectional

    def initial_h(self, batch_size: int, /):
        return torch.zeros(batch_size, self.layers_count * self.bi, self.hidden_dim, device=Device)

    def initial_c(self, batch_size: int, /):
        return torch.zeros(batch_size, self.layers_count * self.bi, self.hidden_dim, device=Device)


class Encoder(RNN):
    def __init__(self, words_n: int, embed_dim: int, hidden_dim: int, /):
        self.words_n = words_n

        self.embedding = nn.Embedding(
            num_embeddings=words_n,
            embedding_dim=embed_dim,
            padding_idx=Token_PAD,
        )

        super().__init__(embed_dim, hidden_dim, 1, False)

    def __getnewargs__(self, /):
        return self.words_n, self.embed_dims, self.hidden_dim

    def __call__(self, inp: Tensor, h: Tensor = None, c: Tensor = None) -> tuple[Tensor, Tensor, Tensor]:
        batch = inp.size()[0]
        if h is None:
            h = self.initial_h(batch)
        if c is None:
            c = self.initial_c(batch)

        embed = self.embedding(inp)
        out, (h, c) = self.lstm(embed, h, c)
        return out, h, c


class Decoder(RNN):
    def __init__(self, words_n: int, embed_dim: int, hidden_dim: int, /):
        self.words_n = words_n

        self.embedding = nn.Embedding(
            num_embeddings=words_n,
            embedding_dim=embed_dim,
            padding_idx=Token_PAD,
        )

        super().__init__(embed_dim, hidden_dim, 1, False)

        self.linear = nn.Linear(hidden_dim * (self.bi + 1), words_n)

    def __getnewargs__(self, /):
        return self.words_n, self.embed_dims, self.hidden_dim

    def __call__(self, inp: Tensor, h: Tensor, c: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        embed = self.embedding(inp)
        out, (h, c) = self.lstm(embed, h, c)
        out = self.linear(out)
        return out, h, c


class EncoderRNN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, /):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def __call__(self, input: Tensor, hidden: Tensor, /) -> tuple[Tensor, Tensor]:
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def init_hidden(self, /):
        return torch.zeros(1, 1, self.hidden_size, device=Device)


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, /):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def __call__(self, input: Tensor, hidden: Tensor, /) -> tuple[Tensor, Tensor]:
        output = self.embedding(input).view(1, 1, -1)
        output = functional.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def init_hidden(self, /):
        return torch.zeros(1, 1, self.hidden_size, device=Device)


class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size: int, output_size: int, max_length: int = 100, /, dropout_p: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def __call__(self, input: Tensor, hidden: Tensor, encoder_outputs: Tensor, /) -> tuple[Tensor, Tensor, Tensor]:
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = functional.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = functional.relu(output)
        output, hidden = self.gru(output, hidden)

        output = functional.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self, /):
        return torch.zeros(1, 1, self.hidden_size, device=Device)
