from zipfile import ZipFile

import torch
from torch.utils.data import DataLoader

from data import Language, TestDataset, Token_EOS, Token_SOS, token2tensor
from misc import Printer, time
from model import Seq2Seq
from utils import Device, make_deterministic


def evaluate(model: Seq2Seq, ru_lang: Language, en_lang: Language, text_path: str, /,
             data_slice: slice = slice(None)):
    model.eval()

    dataset = TestDataset(
        text_path,
        ru_lang,
        data_slice=data_slice,
    )
    batch = 1
    loader = DataLoader(dataset, batch)

    total = len(dataset)
    with torch.no_grad(), Printer() as printer, open('answer.txt', 'w') as out:
        printer.print(f'Testing: starting...')
        for i, (ru, ru_l) in enumerate(loader, 1):
            en = token2tensor(Token_SOS, batch, en_lang.sentence_length + 1)
            # en (batch, words)
            prediction = model(ru, ru_l, en, 0.)  # teaching must be off
            # predictions (batch, words_n, words)
            prediction = prediction.transpose(0, 2).squeeze(2)
            # predictions (words, words_n)

            answer = []
            for p in prediction:
                top = p.argmax(0).item()
                if top == Token_EOS:
                    break
                answer.append(top)

            out.write(' '.join(en_lang.index2word[i] for i in answer) + '\n')

            printer.print(f'Testing: {i / total:.0%} [{i:,}/{total:,}]')

    with ZipFile('answer.zip', 'w') as z:
        z.write('answer.txt')


@time
def main(data_path: str, text_path: str, /):
    make_deterministic()
    ru_args, en_args, model_data = torch.load(data_path)

    ru_lang = Language(*ru_args)
    en_lang = Language(*en_args)

    model = Seq2Seq.from_data(model_data).to(Device).eval()

    evaluate(model, ru_lang, en_lang, text_path)


if __name__ == '__main__':
    default_text = 'tests/test-100-lines.txt'
    # default_text = 'datasets/yandex/corpus.en_ru.1m.ru'
    default_data = 'data/data.pt'

    from argparse import ArgumentParser

    parser = ArgumentParser(
        usage='Runs model on given text file and save results to answer.zip with answer.txt',
    )
    parser.add_argument(
        'data',
        nargs='?',
        default=default_data,
        help=f'path to data file. Default: {default_data!r}',
    )
    parser.add_argument(
        'text',
        nargs='?',
        default=default_text,
        help=f'path to text file. Default: {default_text!r}',
    )

    args = parser.parse_args()

    main(args.data, args.text)
