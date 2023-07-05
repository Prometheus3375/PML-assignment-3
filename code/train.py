from itertools import chain
from math import inf

import torch
from numpy import random
from torch import Tensor
from torch.nn import Module, NLLLoss
from torch.optim import Adam, Optimizer

import hyper
from data import EOS, Language, ParaCrawl, Token_EOS, Token_SOS, Yandex
from device import Device
from misc import Printer, Timer, time
from model import AttnDecoderRNN, EncoderRNN


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


@time
def main():
    # region Fix random
    torch.manual_seed(0)
    random.seed(0)
    # endregion

    # region Prepare data
    with Timer('Data preparation time: %s'):
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

        all_data = yandex, paracrawl,

        for data in all_data:
            ru_lang.add_sentences(data.ru)
            en_lang.add_sentences(data.en)

        dropped = ru_lang.drop_words(hyper.infrequent_words_percentage)

        print(f'{len(dropped)} infrequent words are dropped')

        max_input_length = -inf
        for data in all_data:
            for sentence in data.ru:
                c = sentence.count(' ') + 1
                if c > max_input_length:
                    max_input_length = c
    # endregion

    # region Models and optimizers
    encoder = EncoderRNN(ru_lang.words_n, hyper.hidden_state_size).to(Device).train()
    decoder = AttnDecoderRNN(hyper.hidden_state_size, en_lang.words_n, max_input_length).to(Device).train()

    encoder_optimizer = Adam(encoder.parameters(), lr=hyper.learning_rate)
    decoder_optimizer = Adam(decoder.parameters(), lr=hyper.learning_rate)
    criterion = NLLLoss()
    # endregion

    # region Training
    processed = 0
    total = sum(len(d) for d in all_data)
    log_interval = hyper.log_interval
    with Printer() as printer:
        printer.print(f'Training: starting...')
        for i, (ru, en) in enumerate(chain(*all_data), 1):
            ru = ru_lang.sentence2tensor(f'{ru} {EOS}').view(-1, 1)
            en = en_lang.sentence2tensor(f'{en} {EOS}').view(-1, 1)

            train(ru, en, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

            processed += 1

            if i % log_interval == 0:
                printer.print(f'Training: {processed / total:.0%} [{processed:,}/{total:,}]')

            # print_loss_total += loss
            # plot_loss_total += loss
            #
            # if iter % print_every == 0:
            #     print_loss_avg = print_loss_total / print_every
            #     print_loss_total = 0
            #     print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
            #                                  iter, iter / n_iters * 100, print_loss_avg))
            #
            # if iter % plot_every == 0:
            #     plot_loss_avg = plot_loss_total / plot_every
            #     plot_losses.append(plot_loss_avg)
            #     plot_loss_total = 0

        printer.print(f'Training: completed')
    # endregion

    torch.save(
        (
            hyper.hidden_state_size,
            max_input_length,
            ru_lang.word_counter,
            en_lang.word_counter,
            encoder.cpu().eval().state_dict(),
            decoder.cpu().eval().state_dict(),
        ),
        'data.pth',
    )


if __name__ == '__main__':
    main()
