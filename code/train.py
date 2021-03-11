from math import ceil

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import ConcatDataset, DataLoader

import H
from data import Language, ParaCrawl, Token_PAD, Yandex
from misc import Printer, Timer, time
from model import Attention, Decoder, Encoder, Seq2Seq
from test import evaluate
from utils import Device, make_deterministic


def print_tensor(t: Tensor, name: str = 'tensor'):
    print(f'\n{name} of size {tuple(t.size())}\n{t}')


@time
def main():
    make_deterministic()

    # region Prepare data
    with Timer('\nData preparation time: %s\n'):
        ru_lang = Language()
        en_lang = Language()

        yandex = Yandex(
            'datasets/yandex/corpus.en_ru.1m.ru',
            'datasets/yandex/corpus.en_ru.1m.en',
            ru_lang,
            en_lang,
            data_slice=H.dataset_slice,
        )

        paracrawl = ParaCrawl(
            'datasets/paracrawl/en-ru.txt',
            ru_lang,
            en_lang,
            data_slice=slice(0),
        )

        low = ru_lang.lower_than(H.ru_word_count_minimum)
        infrequent_words_n = max(ceil(ru_lang.words_n * H.infrequent_words_percent), len(low))
        if infrequent_words_n > 0:
            ru_lang.drop_words(ru_lang.lowk(infrequent_words_n))
            print(f'{infrequent_words_n:,} infrequent Russian words are dropped')

        low = en_lang.lower_than(H.en_word_count_minimum)
        if len(low) > 0:
            en_lang.drop_words(*low)
            print(f'{len(low):,} infrequent English words are dropped')

        print(f'Russian language: {ru_lang.words_n:,} words, {ru_lang.sentence_length:,} words in a sentence')
        print(f'English language: {en_lang.words_n:,} words, {en_lang.sentence_length:,} words in a sentence')

        batch = H.batch_size
        dataset = ConcatDataset((yandex, paracrawl))
        loader = DataLoader(dataset, batch, shuffle=True)
    # endregion

    # region Models and optimizers
    model = Seq2Seq(
        Encoder(ru_lang.words_n, H.encoder_embed_dim, H.encoder_hidden_dim, H.encoder_bi, H.decoder_hd),
        Attention(H.encoder_hd, H.decoder_hd),
        Decoder(en_lang.words_n, H.decoder_embed_dim, H.decoder_hidden_dim, H.encoder_hd),
    ).to(Device).train()

    optimizer = Adam(model.parameters(), lr=H.learning_rate)
    criterion = CrossEntropyLoss(ignore_index=Token_PAD, reduction='sum')
    # endregion

    # region Training
    teaching_percent = H.teaching_percent
    total = len(dataset)
    log_interval = max(5, round(total / batch / 100))

    for epoch in range(1, H.epochs + 1):
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
                clip_grad_norm_(model.parameters(), H.gradient_norm_clip)
                optimizer.step()

                # Print log
                if i % log_interval == 0:
                    printer.print(f'Train epoch {epoch}: {i * batch / total:.1%} [{i * batch:,}/{total:,}]')

            printer.print(f'Train epoch {epoch}: completed')
    # endregion

    torch.save(
        (
            ru_lang.__getnewargs__(),
            en_lang.__getnewargs__(),
            model.cpu().eval().data,
        ),
        'data/data.pt',
    )

    evaluate(model.to(Device), ru_lang, en_lang, 'datasets/yandex/corpus.en_ru.1m.ru',
             slice(H.dataset_slice.stop + 1, H.dataset_slice.stop + 1 + 100))


if __name__ == '__main__':
    main()
