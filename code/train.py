from math import ceil

import torch
from numpy import random
from torch import Tensor
from torch.nn import CrossEntropyLoss, Module
from torch.optim import Adam, Optimizer
from torch.utils.data import ConcatDataset, DataLoader

import hyper
from data import Language, ParaCrawl, Token_EOS, Token_SOS, Yandex, token2tensor
from device import Device
from misc import Printer, Timer, time
from model import AttnDecoderRNN, Decoder, Encoder, EncoderRNN


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


def print_tensor(name: str, t: Tensor):
    print(f'\n{name} of size {tuple(t.size())}\n{t}')


@time
def main():
    # region Fix random
    torch.manual_seed(0)
    random.seed(0)
    # endregion

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
    criterion = CrossEntropyLoss()
    # endregion

    # region Training
    sos = token2tensor(Token_SOS, batch)
    eos = token2tensor(Token_EOS, batch)

    processed = 0
    total = len(dataset)
    log_interval = hyper.log_interval

    with Printer() as printer:
        printer.print(f'Training: starting...')
        for i, (ru, en) in enumerate(loader, 1):
            # print_tensor('ru', ru)
            # print_tensor('en', en)

            # Zero the parameter gradients
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # Run data through coders
            encoder_inp = torch.cat((ru, eos), dim=1)
            # print_tensor('encoder_inp', encoder_inp)
            encoded, hc = encoder(encoder_inp)

            decoder_inp = torch.cat((sos, en), dim=1)
            decoder_out = torch.cat((en, eos), dim=1)
            # print_tensor('decoder_inp', decoder_inp)
            # print_tensor('decoder_out', decoder_out)
            decoded, hc = decoder(decoder_inp, hc)
            # print_tensor('decoded', decoded)

            loss = criterion(decoded, decoder_out)

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
