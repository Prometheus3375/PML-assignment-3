from collections import OrderedDict
from typing import final

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from data import Token_PAD
from utils import Device

_tt = tuple[Tensor, Tensor]


class Model(nn.Module):
    def __getattr__(self, attr: str, /):
        # details: https://github.com/pytorch/pytorch/issues/13981
        try:
            return super().__getattr__(attr)
        except AttributeError:
            return self.__getattribute__(attr)

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
        return torch.zeros(self.layers_count * (self.bi + 1), batch_size, self.hidden_dim, device=Device)

    def initial_c(self, batch_size: int, /):
        return torch.zeros(self.layers_count * (self.bi + 1), batch_size, self.hidden_dim, device=Device)

    def initial_hc(self, batch_size: int, /):
        return self.initial_h(batch_size), self.initial_c(batch_size)


@final
class Encoder(RNN):
    def __init__(self, words_n: int, embed_dim: int, hidden_dim: int, /):
        super().__init__(embed_dim, hidden_dim, 1, False)

        self.words_n = words_n

        self.embedding = nn.Embedding(
            num_embeddings=words_n,
            embedding_dim=embed_dim,
            padding_idx=Token_PAD,
        )

    def __getnewargs__(self, /):
        return self.words_n, self.input_dim, self.hidden_dim

    def __call__(self, inp: _tt, hc: _tt = None) -> tuple[_tt, _tt]:
        data, lengths = inp
        b, l = data.size()

        if hc is None:
            hc = self.initial_hc(b)

        embed = self.embedding(data)
        seqs = pack_padded_sequence(embed, lengths, True, False)
        out, hc = self.lstm(seqs, hc)
        out = pad_packed_sequence(out, True, Token_PAD, l)
        return out, hc


@final
class Decoder(RNN):
    def __init__(self, words_n: int, embed_dim: int, hidden_dim: int, /):
        super().__init__(embed_dim, hidden_dim, 1, False)

        self.words_n = words_n

        self.embedding = nn.Embedding(
            num_embeddings=words_n,
            embedding_dim=embed_dim,
            padding_idx=Token_PAD,
        )

        self.linear = nn.Linear(hidden_dim * (self.bi + 1), words_n)

    def __getnewargs__(self, /):
        return self.words_n, self.input_dim, self.hidden_dim

    def __call__(self, inp: _tt, hc: _tt) -> tuple[_tt, _tt]:
        data, lengths = inp
        l = data.size()[1]
        embed = self.embedding(data)
        seqs = pack_padded_sequence(embed, lengths, True, False)
        out, hc = self.lstm(seqs, hc)
        out, lengths = pad_packed_sequence(out, True, Token_PAD, l)

        out = self.linear(out)
        b, seq, cls = out.size()
        out = out.view(b, cls, seq)
        return out, hc


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
