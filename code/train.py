from math import ceil

import torch
from numpy import random
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Adam, Optimizer
from torch.utils.data import ConcatDataset, DataLoader

import hyper
from data import Language, ParaCrawl, Token_EOS, Token_PAD, Token_SOS, Yandex
from misc import Printer, Timer, time
from model import AttnDecoderRNN, Decoder, Encoder, EncoderRNN
from utils import Device, fix_random


def train(
        input_tensor: Tensor,
        target_tensor: Tensor,
        encoder: EncoderRNN,
        decoder: AttnDecoderRNN,
        encoder_optimizer: Optimizer,
        decoder_optimizer: Optimizer,
        criterion: Module,
):
    encoder_hidden = encoder.init_hidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size()[0]
    target_length = target_tensor.size()[0]

    encoder_outputs = torch.zeros(decoder.max_length, encoder.hidden_size, device=Device)

    loss: Tensor = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[Token_SOS]], device=Device)

    decoder_hidden = encoder_hidden

    teach = (
            hyper.teaching_percentage == 1 or
            (hyper.teaching_percentage != 0 and random.rand() <= hyper.teaching_percentage)
    )

    for di in range(target_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
        loss += criterion(decoder_output, target_tensor[di])

        if teach:
            decoder_input = target_tensor[di]
        else:
            topi: Tensor
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input

            if decoder_input.item() == Token_EOS:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def print_tensor(t: Tensor, name: str = 'tensor'):
    print(f'\n{name} of size {tuple(t.size())}\n{t}')


@time
def main():
    fix_random()

    # region Prepare data
    with Timer('\nData preparation time: %s\n'):
        ru_lang = Language()
        en_lang = Language()

        yandex = Yandex(
            'datasets/yandex/corpus.en_ru.1m.ru',
            'datasets/yandex/corpus.en_ru.1m.en',
            ru_lang,
            en_lang,
            data_slice=hyper.dataset_slice,
        )

        paracrawl = ParaCrawl(
            'datasets/paracrawl/en-ru.txt',
            ru_lang,
            en_lang,
            data_slice=slice(0),
        )

        infrequent_words_n = ceil(ru_lang.words_n * hyper.infrequent_words_percentage)
        ru_lang.drop_words(ru_lang.lowk(infrequent_words_n))
        print(f'{infrequent_words_n:,} infrequent Russian words are dropped')

        print(f'Russian language: {ru_lang.words_n:,} words, {ru_lang.sentence_length:,} words in a sentence')
        print(f'English language: {en_lang.words_n:,} words, {en_lang.sentence_length:,} words in a sentence')

        batch = hyper.batch_size
        dataset = ConcatDataset((yandex, paracrawl))
        loader = DataLoader(dataset, batch, shuffle=True)
    # endregion

    # region Models and optimizers
    encoder = Encoder(ru_lang.words_n, hyper.embed_dim, hyper.hidden_dim).to(Device).train()
    decoder = Decoder(en_lang.words_n, hyper.embed_dim, hyper.hidden_dim).to(Device).train()

    encoder_optimizer = Adam(encoder.parameters(), lr=hyper.learning_rate)
    decoder_optimizer = Adam(decoder.parameters(), lr=hyper.learning_rate)
    criterion = CrossEntropyLoss(ignore_index=Token_PAD)
    # endregion

    # region Training

    processed = 0
    total = len(dataset)
    log_interval = hyper.log_interval

    with Printer() as printer:
        printer.print(f'Training: starting...')
        for i, (ru_eos, en_sos, en_eos) in enumerate(loader, 1):
            # print_tensor(ru_eos, 'ru_eos')
            # print_tensor(en_sos, 'en_sos')
            # print_tensor(en_eos, 'en_eos')

            # Zero the parameter gradients
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # Run data through coders
            encoded, hc = encoder(ru_eos)
            decoded, hc = decoder(en_sos, hc)
            # print_tensor(decoded, 'decoded')

            loss = criterion(decoded, en_eos)

            # Back propagate and perform optimization
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()

            # Print log
            processed += batch
            if i % log_interval == 0:
                printer.print(f'Training: {processed / total:.1%} [{processed:,}/{total:,}]')

        printer.print(f'Training: completed')
    # endregion

    torch.save(
        (
            ru_lang.__getnewargs__(),
            en_lang.__getnewargs__(),
            encoder.cpu().eval().data,
            decoder.cpu().eval().data,
        ),
        'data/data.pth',
    )


if __name__ == '__main__':
    main()
