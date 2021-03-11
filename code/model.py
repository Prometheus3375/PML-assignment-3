from collections import OrderedDict
from typing import final

import torch
from numpy import random
from torch import Tensor, nn
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
    def __init__(self, input_dim: int, hidden_dim: int, /,
                 layers_count: int = 1, bi: bool = False, batch_first: bool = False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=layers_count,
            bidirectional=bi,
            batch_first=batch_first,
        )

    def __getnewargs__(self, /):
        return self.input_dim, self.hidden_dim, self.layers_count, self.bi, self.batch_first

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

    @property
    def batch_first(self, /):
        return self.lstm.batch_first

    def initial_h(self, batch_size: int, /):
        return torch.zeros(self.layers_count * (self.bi + 1), batch_size, self.hidden_dim, device=Device)

    def initial_c(self, batch_size: int, /):
        return torch.zeros(self.layers_count * (self.bi + 1), batch_size, self.hidden_dim, device=Device)

    def initial_hc(self, batch_size: int, /):
        return self.initial_h(batch_size), self.initial_c(batch_size)


@final
class Encoder(RNN):
    def __init__(self, words_n: int, embed_dim: int, hidden_dim: int, /):
        super().__init__(embed_dim, hidden_dim, bi=True)

        self.words_n = words_n

        self.embedding = nn.Embedding(
            num_embeddings=words_n,
            embedding_dim=embed_dim,
            padding_idx=Token_PAD,
        )

        self.linear_h = nn.Linear(hidden_dim * (self.bi + 1), hidden_dim)
        self.linear_c = nn.Linear(hidden_dim * (self.bi + 1), hidden_dim)

    def __getnewargs__(self, /):
        return self.words_n, self.input_dim, self.hidden_dim

    def _process_v(self, v: Tensor, fc: nn.Linear) -> Tensor:
        # v[-2, :, :]  last forward
        # v[-1, :, :]  last backward
        v = torch.cat((v[-2, :, :], v[-1, :, :]), dim=1)
        # v (batch, hidden_dim * 2)
        v = self.linear_h(v)
        # v (batch, hidden_dim)
        v = torch.tanh(v)
        v = v.unsqueeze(0)
        # v (1, batch, hidden_dim)
        return v

    def __call__(self, data: Tensor, lengths: Tensor, hc: _tt = None) -> tuple[Tensor, _tt]:
        l, b = data.size()

        if hc is None:
            hc = self.initial_hc(b)

        # data (words, batch)
        embed = self.embedding(data)
        # embed (words, batch, input_dim)
        seqs = pack_padded_sequence(embed, lengths, self.batch_first, False)
        out, hc = self.lstm(seqs, hc)
        out, lengths = pad_packed_sequence(out, self.batch_first, Token_PAD, l)
        # out (words, batch, (bi + 1) * hidden_dim)
        # hc (layers_count * (bi + 1), batch, hidden_dim)
        h, c = hc
        h = self._process_v(h, self.linear_h)
        c = self._process_v(c, self.linear_c)
        # hc (1, batch, hidden_dim)
        return out, (h, c)


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

    def __call__(self, data: Tensor, hc: _tt) -> tuple[Tensor, _tt]:
        # data (batch)
        embed = self.embedding(data)
        # embed (batch, input_dim)
        embed = embed.unsqueeze(0)
        # embed (words = 1, batch, input_dim)
        out, hc = self.lstm(embed, hc)
        # out (words = 1, batch, (bi + 1) * hidden_dim)
        # hc (layers_count * (bi + 1), batch, hidden_dim)

        out = self.linear(out)
        # out (words = 1, batch, words_n)
        out = out.squeeze(0)
        # out (batch, words_n)
        return out, hc


@final
class Seq2Seq(Model):
    def __init__(self, encoder: Encoder, decoder: Decoder, /):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def __getnewargs__(self, /):
        return self.encoder, self.decoder

    @property
    def data(self, /):
        return self.encoder.data, self.decoder.data

    @classmethod
    def from_data(cls, data: tuple[tuple, tuple], /):
        encoder, decoder = data
        return cls(Encoder.from_data(encoder), Decoder.from_data(decoder))

    @staticmethod
    def teach(teaching_percent: float, /):
        return (
                teaching_percent >= 1 or
                (teaching_percent > 0 and random.rand() < teaching_percent)
        )

    def __call__(self, inp: Tensor, inp_lengths: Tensor, out: Tensor, teaching_percent: float, /) -> Tensor:
        # (batch, words) -> (words, batch)
        inp = inp.transpose(0, 1)
        out = out.transpose(0, 1)

        encoded, hc = self.encoder(inp, inp_lengths)

        l, b = out.size()
        predictions = torch.zeros(l, b, self.decoder.words_n, device=Device) + Token_PAD
        inp: Tensor = out[0]
        # inp (batch)
        for k in range(l - 1):
            predict, hc = self.decoder(inp, hc)
            predictions[k] = predict
            inp = out[k + 1] if self.teach(teaching_percent) else predict.argmax(1)

        predict, hc = self.decoder(inp, hc)
        predictions[-1] = predict
        # predictions (words, batch, words_n)

        predictions = predictions.permute(1, 2, 0)
        # predictions (batch, words_n, words)
        return predictions
