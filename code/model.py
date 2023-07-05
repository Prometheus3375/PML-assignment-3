import torch
import torch.nn.functional as F
from torch import Tensor, nn

from device import Device


class Encoder(nn.Module):
    def __init__(self, language_: int):
        pass


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
        output = F.relu(output)
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

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def init_hidden(self, /):
        return torch.zeros(1, 1, self.hidden_size, device=Device)
