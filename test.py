from zipfile import ZipFile

import torch
from torch import Tensor

from data import EOS, Language, Token_EOS, Token_SOS, preprocess_ru
from device import Device
from model import AttnDecoderRNN, EncoderRNN


def evaluate(encoder: EncoderRNN, decoder: AttnDecoderRNN, input_tensor: Tensor):
    with torch.no_grad():
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(decoder.max_length, encoder.hidden_size, device=Device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[Token_SOS]], device=Device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_indexes = []

        for di in range(decoder.max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == Token_EOS:
                break

            decoded_indexes.append(topi.item())
            decoder_input = topi.squeeze().detach()

        return decoded_indexes


if __name__ == '__main__':
    default_model = 'data.pth'

    from argparse import ArgumentParser

    parser = ArgumentParser(
        usage='Runs model on test.ru.txt and save results to answer.zip with answer.txt',
    )
    parser.add_argument(
        'model',
        help=f'path to data file. Default: {default_model!r}',
        default=default_model,
    )

    args = parser.parse_args()

    hidden_state_size, max_output_length, ru_counter, en_counter, encoder_w, decoder_w = torch.load(args.model)

    ru_lang = Language(ru_counter)
    en_lang = Language(en_counter)

    encoder = EncoderRNN(ru_lang.words_n, hidden_state_size)
    encoder.load_state_dict(encoder_w)
    encoder.to(Device).eval()

    decoder = AttnDecoderRNN(hidden_state_size, en_lang.words_n, max_output_length)
    decoder.load_state_dict(decoder_w)
    decoder.to(Device).eval()

    with open('test.ru.txt') as f:
        with open('answer.txt', 'w') as out:
            for line in f:
                sentence = preprocess_ru(line)
                ru = ru_lang.sentence2tensor(f'{sentence} {EOS}').view(-1, 1)

                answer = evaluate(encoder, decoder, ru)
                out.write(' '.join(en_lang.index2word[i] for i in answer) + '\n')

    with ZipFile('answer.zip', 'w') as z:
        z.write('answer.txt')
