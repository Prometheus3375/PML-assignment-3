from math import ceil

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import ConcatDataset, DataLoader

import hyper
from data import Language, ParaCrawl, Token_PAD, Yandex
from misc import Printer, Timer, time
from model import Decoder, Encoder, Seq2Seq
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

        low = ru_lang.lower_than(2)
        infrequent_words_n = max(ceil(ru_lang.words_n * hyper.infrequent_words_percent), len(low))
        ru_lang.drop_words(ru_lang.lowk(infrequent_words_n))
        print(f'{infrequent_words_n:,} infrequent Russian words are dropped')

        low = en_lang.lower_than(2)
        en_lang.drop_words(*low)
        print(f'{len(low):,} infrequent English words are dropped')

        print(f'Russian language: {ru_lang.words_n:,} words, {ru_lang.sentence_length:,} words in a sentence')
        print(f'English language: {en_lang.words_n:,} words, {en_lang.sentence_length:,} words in a sentence')

        batch = hyper.batch_size
        dataset = ConcatDataset((yandex, paracrawl))
        loader = DataLoader(dataset, batch, shuffle=True)
    # endregion

    # region Models and optimizers
    encoder = Encoder(ru_lang.words_n, hyper.embed_dim, hyper.hidden_dim).to(Device).train()
    decoder = Decoder(en_lang.words_n, hyper.embed_dim, hyper.hidden_dim).to(Device).train()
    model = Seq2Seq(encoder, decoder)

    optimizer = Adam(model.parameters(), lr=hyper.learning_rate)
    criterion = CrossEntropyLoss(ignore_index=Token_PAD, reduction='sum')
    # endregion

    # region Training
    teaching_percent = hyper.teaching_percent
    total = len(dataset)
    log_interval = hyper.log_interval

    for epoch in range(1, hyper.epochs + 1):
        processed = 0
        with Printer() as printer:
            printer.print(f'Train epoch {epoch}: starting...')
            for i, ((ru, ru_l), en_sos, en_eos) in enumerate(loader, 1):
                # Zero the parameter gradients
                optimizer.zero_grad()
                # Run data through model
                predictions = model(ru, ru_l, en_sos, teaching_percent)
                # Calculate loss
                loss = criterion(predictions, en_eos)
                # Back propagate and perform optimization
                loss.backward()
                optimizer.step()

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
            model.cpu().eval().data,
        ),
        'data/data.pth',
    )


if __name__ == '__main__':
    main()
