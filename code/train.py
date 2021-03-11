from math import ceil

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import ConcatDataset, DataLoader

import hyper
from data import Language, ParaCrawl, Token_PAD, Yandex
from misc import Printer, Timer, time
from model import Decoder, Encoder
from utils import Device, make_determenistic


def print_tensor(t: Tensor, name: str = 'tensor'):
    print(f'\n{name} of size {tuple(t.size())}\n{t}')


@time
def main():
    make_determenistic()

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
    criterion = CrossEntropyLoss(ignore_index=Token_PAD, reduction='sum')
    # endregion

    # region Training
    total = len(dataset)
    log_interval = hyper.log_interval
    for epoch in range(1, hyper.epochs + 1):
        processed = 0
        with Printer() as printer:
            printer.print(f'Train epoch {epoch}: starting...')
            for i, (ru_eos_t, en_sos_t, en_eos) in enumerate(loader, 1):
                # print_tensor(ru_eos_t[0], 'ru_eos')
                # print_tensor(en_sos_t[0], 'en_sos')
                # print_tensor(en_eos, 'en_eos')

                # Zero the parameter gradients
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                # Run data through coders
                encoded, hc = encoder(ru_eos_t)
                decoded, hc = decoder(en_sos_t, hc)
                # print_tensor(decoded, 'decoded')

                loss = criterion(decoded, en_eos)

                # Back propagate and perform optimization
                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()

                # Print log
                processed += batch
                if i % log_interval == 0:
                    printer.print(f'Train epoch {epoch}: {processed / total:.1%} [{processed:,}/{total:,}]')

            printer.print(f'Train epoch {epoch}: completed')
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
