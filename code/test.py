from zipfile import ZipFile

import torch
from torch.utils.data import DataLoader

from data import Language, TestDataset, Token_EOS, Token_SOS, token2tensor
from misc import Printer, time
from model import Decoder, Encoder
from utils import Device, fix_random


@time
def main(text_path: str, data_path: str):
    fix_random()

    ru_args, en_args, encoder_data, decoder_data = torch.load(data_path)

    ru_lang = Language(*ru_args)
    en_lang = Language(*en_args)

    dataset = TestDataset(text_path, ru_lang)
    batch = 1
    loader = DataLoader(dataset, batch)

    encoder = Encoder.from_data(encoder_data).to(Device).eval()
    decoder = Decoder.from_data(decoder_data).to(Device).eval()

    total = len(dataset)
    with torch.no_grad(), Printer() as printer, open('answer.txt', 'w') as out:
        printer.print(f'Testing: starting...')
        for i, sentence in enumerate(loader, 1):
            # print_tensor(sentence, 'encoder_inp')

            encoded, hc = encoder(sentence)

            decoder_inp = token2tensor(Token_SOS, batch)
            # print_tensor(decoder_inp, 'decoder_inp')

            answer = []
            for _ in range(en_lang.sentence_length):
                decoded, hc = decoder(decoder_inp, hc)

                topv, topi = decoded.topk(1, dim=1)
                token = topi.item()
                if token == Token_EOS:
                    break

                answer.append(token)
                decoder_inp = token2tensor(token, batch)
                # print_tensor(decoder_inp, 'decoder_inp')

            out.write(' '.join(en_lang.index2word[i] for i in answer) + '\n')

            printer.print(f'Testing: {i / total:.0%} [{i:,}/{total:,}]')

    with ZipFile('answer.zip', 'w') as z:
        z.write('answer.txt')


if __name__ == '__main__':
    default_text = 'tests/test-100-lines.txt'
    default_data = 'data/data.pth'

    from argparse import ArgumentParser

    parser = ArgumentParser(
        usage='Runs model on given text file and save results to answer.zip with answer.txt',
    )
    parser.add_argument(
        'text',
        nargs='?',
        default=default_text,
        help=f'path to text file. Default: {default_text!r}',
    )
    parser.add_argument(
        'data',
        nargs='?',
        default=default_data,
        help=f'path to data file. Default: {default_data!r}',
    )

    args = parser.parse_args()

    main(args.text, args.data)
