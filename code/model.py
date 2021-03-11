from collections import OrderedDict
from typing import final

import torch
from numpy import random
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from data import Token_PAD
from utils import Device


class Model(nn.Module):
    def __getattr__(self, attr: str, /):
        # details: https://github.com/pytorch/pytorch/issues/13981
        try:
            return super().__getattr__(attr)
        except AttributeError:
            pass

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
        self.rnn = nn.GRU(
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
        return self.rnn.input_size

    @property
    def hidden_dim(self, /):
        return self.rnn.hidden_size

    @property
    def layers_count(self, /):
        return self.rnn.num_layers

    @property
    def bi(self, /):
        return self.rnn.bidirectional

    @property
    def batch_first(self, /):
        return self.rnn.batch_first


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
        self.dropout = nn.Dropout()

        self.linear = nn.Linear(hidden_dim * (self.bi + 1), hidden_dim)

    def __getnewargs__(self, /):
        return self.words_n, self.input_dim, self.hidden_dim

    def __call__(self, data: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        l, b = data.size()

        # data (words, batch)
        embed = self.embedding(data)
        # embed (words, batch, input_dim)
        embed = self.dropout(embed)
        seqs = pack_padded_sequence(embed, lengths, self.batch_first, False)
        out, h = self.rnn(seqs)
        out, lengths = pad_packed_sequence(out, self.batch_first, Token_PAD, l)
        # out (words, batch, (bi + 1) * hidden_dim)
        # h (layers_count * (bi + 1), batch, hidden_dim)
        # h[0] last forward
        # h[1] last backward
        h = torch.cat((h[0], h[1]), dim=1)
        # h (batch, hidden_dim * 2)
        h = self.linear(h)
        # v (batch, hidden_dim)
        h = torch.tanh(h)
        h = h.unsqueeze(0)
        # h (1, batch, hidden_dim)
        out = out.transpose(0, 1)
        # out (batch, words, (bi + 1) * hidden_dim)
        return out, h


@final
class Attention(Model):
    def __init__(self, hidden_dim: int, /):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        self.softmax = nn.Softmax(1)

    def __getnewargs__(self, /):
        return self.hidden_dim,

    def __call__(self, encoder_out: Tensor, mask: Tensor, v: Tensor, /):
        # v (1, batch, hidden_dim)
        # encoder_out (batch, words, 2 * hidden_dim)

        l = encoder_out.size()[1]

        v = v.transpose(0, 1)
        # v (batch, 1, hidden_dim)
        v = v.repeat(1, l, 1)
        # v (batch, words, hidden_dim)

        en = torch.cat((v, encoder_out), dim=2)
        # en (batch, words, hidden_dim * 3)
        en = self.attn(en)
        # en (batch, words, hidden_dim)
        en = torch.tanh(en)

        attention = self.v(en)
        # attention (batch, words, 1)
        attention = attention.squeeze(2)
        # attention (batch, words)
        attention.masked_fill_(mask == 0, -1e10)
        attention = self.softmax(attention)

        return attention


@final
class Decoder(RNN):
    def __init__(self, words_n: int, embed_dim: int, hidden_dim: int, /):
        super().__init__(embed_dim + hidden_dim * 2, hidden_dim, 1, False)

        self.words_n = words_n

        self.embedding = nn.Embedding(
            num_embeddings=words_n,
            embedding_dim=embed_dim,
            padding_idx=Token_PAD,
        )
        self.dropout = nn.Dropout()

        self.linear = nn.Linear(hidden_dim * 2 + hidden_dim * (self.bi + 1) + embed_dim, words_n)

    def __getnewargs__(self, /):
        return self.words_n, self.input_dim - self.hidden_dim * 2, self.hidden_dim

    def __call__(self, data: Tensor, weighted: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
        # weighted (1, batch, hidden_dim * 2)
        # data (batch)
        data = data.unsqueeze(0)
        # data (words = 1, batch)
        embed = self.embedding(data)
        # embed (words = 1, batch, input_dim)
        embed = self.dropout(embed)

        inp = torch.cat((embed, weighted), dim=2)
        # embed (words = 1, batch, input_dim + hidden_dim * 2)
        out, h = self.rnn(inp, h)
        # out (words = 1, batch, (bi + 1) * hidden_dim)
        # h (layers_count * (bi + 1), batch, hidden_dim)

        result = torch.cat((out, weighted, embed), dim=2)
        # result (words = 1, batch, (bi + 1) * hidden_dim + hidden_dim * 2 + input_dim)
        result = self.linear(result)
        # result (words = 1, batch, words_n)
        result = result.squeeze(0)
        # result (batch, words_n)
        return result, h


@final
class Seq2Seq(Model):
    def __init__(self, encoder: Encoder, attention: Attention, decoder: Decoder, /):
        super().__init__()
        self.encoder = encoder
        self.attention = attention
        self.decoder = decoder

    def __getnewargs__(self, /):
        return self.encoder, self.attention, self.decoder

    @property
    def data(self, /):
        return self.encoder.data, self.attention.data, self.decoder.data

    @classmethod
    def from_data(cls, data: tuple[tuple, tuple, tuple], /):
        encoder, attention, decoder = data
        return cls(Encoder.from_data(encoder), Attention.from_data(attention), Decoder.from_data(decoder))

    @staticmethod
    def teach(teaching_percent: float, /):
        return (
                teaching_percent >= 1 or
                (teaching_percent > 0 and random.rand() < teaching_percent)
        )

    def _eval_weighted(self, encoder_out: Tensor, mask: Tensor, v: Tensor) -> Tensor:
        a = self.attention(encoder_out, mask, v)
        # a (batch, words)
        a = a.unsqueeze(1)
        # attention (batch, 1, words)
        weighted = torch.bmm(a, encoder_out)
        # weighted (batch, 1, 2 * hidden_dim)
        weighted = weighted.transpose(0, 1)
        # weighted (1, batch, 2 * hidden_dim)
        return weighted

    def __call__(self, inp: Tensor, inp_lengths: Tensor, out: Tensor, teaching_percent: float, /) -> Tensor:
        mask = inp != Token_PAD
        # mask (batch, words)

        # (batch, words) -> (words, batch)
        inp = inp.transpose(0, 1)
        out = out.transpose(0, 1)

        encoded, h = self.encoder(inp, inp_lengths)
        # encoded (batch, words, 2 * hidden_dim)

        l, b = out.size()
        predictions = torch.zeros(l, b, self.decoder.words_n, device=Device) + Token_PAD
        inp = out[0]
        # inp (batch)
        weighted = self._eval_weighted(encoded, mask, h)
        # weighted (1, batch, 2 * hidden_dim)
        for k in range(l - 1):
            predict, h = self.decoder(inp, weighted, h)
            predictions[k] = predict

            inp = out[k + 1] if self.teach(teaching_percent) else predict.argmax(1)
            weighted = self._eval_weighted(encoded, mask, h)

        predict, h = self.decoder(inp, weighted, h)
        predictions[-1] = predict
        # predictions (words, batch, words_n)

        predictions = predictions.permute(1, 2, 0)
        # predictions (batch, words_n, words)
        return predictions
