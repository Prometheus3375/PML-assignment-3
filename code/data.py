import re
from collections import Counter
from math import inf
from typing import TextIO, final

from torch import Tensor, int64, tensor, zeros
from torch.utils.data import Dataset

from utils import Device

Sentence = str

Pattern_number = re.compile(r'-?[\d.,:/]+[$%]?')
Pattern_en_postfixes = re.compile(r"(\w+)['’](m|re|s|ve|d)")

Pattern_ignored = re.compile(r"[^\w\s.,:/$%'’-]+")
Pattern_not_alpha = re.compile(r'\W+')
Pattern_spaces = re.compile(r'\s+')


def preprocess_en(s: str, /) -> Sentence:
    s = s.rstrip().lower()
    s = Pattern_ignored.sub('', s)
    words = s.split()
    for i, w in enumerate(words):
        w = w.strip(",.:'’")

        if Pattern_number.fullmatch(w):
            w = Pattern_not_alpha.sub('', w)
        elif m := Pattern_en_postfixes.fullmatch(w):
            w = m.group(1)
        elif w.endswith("n't"):
            if w == "won't":
                w = 'will not'
            elif w == "can't":
                w = 'cannot'
            else:
                w = f'{w[:-3]} not'
        else:
            w = Pattern_not_alpha.sub(' ', w).strip()
            w = Pattern_spaces.sub(' ', w)

        words[i] = w

    return ' '.join(words)


def preprocess_ru(s: str, /) -> Sentence:
    s = s.rstrip().lower()
    s = Pattern_ignored.sub('', s)
    words = s.split()
    for i, w in enumerate(words):
        w = w.strip(",.:'’")

        if Pattern_number.fullmatch(w):
            w = Pattern_not_alpha.sub('', w)
        else:
            w = Pattern_not_alpha.sub(' ', w).strip()
            w = Pattern_spaces.sub(' ', w)

        words[i] = w

    return ' '.join(words)


PAD = '~'
SOS = '^'
EOS = '@'
NIL = '*'
Special = [PAD, SOS, EOS, NIL]
Token_PAD = Special.index(PAD)
Token_SOS = Special.index(SOS)
Token_EOS = Special.index(EOS)
Token_NIL = Special.index(NIL)


def token2tensor(token: int, /, *size: int) -> Tensor:
    return zeros(*size, dtype=int64, device=Device) + token


@final
class Language:
    __slots__ = 'word_counter', '_sentence_length', 'word2index', 'index2word',

    def __init__(self, word_counter: Counter[str] = None, sentence_length: int = 0, /):
        self.word_counter: Counter[str] = word_counter if isinstance(word_counter, Counter) else Counter()
        self._sentence_length = sentence_length
        self._reindex()

    def __getnewargs__(self, /):
        return self.word_counter, self._sentence_length

    def _reindex(self, /):
        self.index2word: list[str] = Special + sorted(self.word_counter)
        self.word2index: dict[str, int] = {w: i for i, w in enumerate(self.index2word)}

    @property
    def words_n(self, /):
        return len(self.word2index)

    @property
    def sentence_length(self, /):
        return self._sentence_length

    def add_sentences(self, sentences: list[Sentence], /):
        for sentence in sentences:
            n = 0
            for word in sentence.split():
                self.word_counter[word] += 1
                n += 1

            if n > self._sentence_length:
                self._sentence_length = n

        self._reindex()

    def lower_than(self, count: int, /):
        return frozenset(word for word, c in self.word_counter.items() if c < count)

    def topk(self, k: int, /):
        return frozenset(sorted(
            self.word_counter,
            key=lambda w: self.word_counter[w],
            reverse=True,
        )[:k])

    def lowk(self, k: int, /):
        return frozenset(sorted(
            self.word_counter,
            key=lambda w: self.word_counter[w],
        )[:k])

    def drop_words(self, /, *words: str):
        for word in words:
            del self.word_counter[word]

        self._reindex()

    def sentence2tensor(self, sentence: Sentence, /,
                        with_sos: bool = False, with_eos: bool = False):
        indexes = [Token_SOS] if with_sos else []
        indexes += [self.word2index.get(w, Token_NIL) for w in sentence.split()]
        if with_eos:
            indexes.append(Token_EOS)

        l = len(indexes)
        indexes += [Token_PAD] * (self._sentence_length - l + with_sos + with_eos)
        return tensor(indexes, device=Device), tensor(l)


@final
class LanguageData:
    __slots__ = 'data', 'lang', 'max_length',

    def __init__(self, data: list[Sentence], lang: Language, /):
        self.data = data
        self.lang = lang

        lang.add_sentences(data)

    def __len__(self, /):
        return len(self.data)

    def __iter__(self, /):
        return iter(self.data)

    def get_both(self, index: int, /):
        return self.lang.sentence2tensor(self.data[index], with_sos=True, with_eos=True)

    def get_sos(self, index: int, /):
        return self.lang.sentence2tensor(self.data[index], with_sos=True)

    def get_eos(self, index: int, /):
        return self.lang.sentence2tensor(self.data[index], with_eos=True)

    def get(self, index: int, /):
        return self.data[index]


class RUENDataset(Dataset):
    def __init__(self, ru: list[str], en: list[str], ru_lang: Language, en_lang: Language, /):
        if len(ru) != len(en):
            raise ValueError(f'different number of sentences: ru ({len(ru):,}) and en ({len(en):,})')

        if ru_lang is en_lang:
            raise ValueError(f'same language object is used for RU and EN')

        self.ru = LanguageData([preprocess_ru(s) for s in ru], ru_lang)
        self.en = LanguageData([preprocess_en(s) for s in en], en_lang)

    def __len__(self, /):
        return len(self.ru)

    def __iter__(self, /):
        return zip(self.ru, self.en)

    def __getitem__(self, index: int, /) -> tuple[tuple[Tensor, Tensor], Tensor, Tensor]:
        return (
            self.ru.get_both(index),
            self.en.get_sos(index)[0],
            self.en.get_eos(index)[0],
        )

    def get(self, index: int, /):
        return self.ru.get(index), self.en.get(index)


def readlines(f: TextIO, /, data_slice: slice):
    start = data_slice.start
    stop = data_slice.stop
    step = data_slice.step
    if step is None:
        step = 1

    if step < 0:
        stop = inf if start is None else start + 1  # + 1 since start element must be included
    elif stop is None:
        stop = inf

    lines = []
    for line in f:
        if len(lines) >= stop:
            break

        lines.append(line)

    return lines[data_slice]


@final
class ParaCrawl(RUENDataset):
    def __init__(self, en_ru_file: str, ru_lang: Language, en_lang: Language, /,
                 data_slice: slice = slice(None)):
        with open(en_ru_file, 'r') as f:
            lines = readlines(f, data_slice)

        en_list = []
        ru_list = []
        for line in lines:
            en, ru = line.split('\t')
            en_list.append(en)
            ru_list.append(ru)

        super().__init__(ru_list, en_list, ru_lang, en_lang)


@final
class Yandex(RUENDataset):
    def __init__(self, ru_file: str, en_file: str, ru_lang: Language, en_lang: Language, /,
                 data_slice: slice = slice(None)):
        with open(ru_file) as f:
            ru_list = readlines(f, data_slice)

        with open(en_file) as f:
            en_list = readlines(f, data_slice)

        super().__init__(ru_list, en_list, ru_lang, en_lang)


@final
class TestDataset(Dataset):
    def __init__(self, file: str, lang: Language, /, data_slice: slice = slice(None)):
        with open(file) as f:
            lines = readlines(f, data_slice)

        self.sentences = [preprocess_ru(s) for s in lines]
        self.lang = lang

    def __len__(self, /):
        return len(self.sentences)

    def __iter__(self, /):
        return iter(self.sentences)

    def __getitem__(self, index: int, /):
        return self.lang.sentence2tensor(self.sentences[index], with_eos=True, with_sos=True)

    def get(self, index: int, /):
        return self.sentences[index]
