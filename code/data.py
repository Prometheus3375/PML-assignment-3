import re
from collections import Counter
from typing import final

from torch import tensor
from torch.utils.data import Dataset

from device import Device

Sentence = str

Pattern_number = re.compile(r'[\d.,:/]+[$%]?')
Pattern_en_postfixes = re.compile(r"(\w+)'(m|re|s|ve|d)")

Pattern_ignored = re.compile(r"[^\w.,:/$%-' ]+")
Pattern_not_alpha = re.compile(r'\W+')
Pattern_spaces = re.compile(r'\s+')


def preprocess_en(s: str, /) -> Sentence:
    s = s.rstrip().lower()
    s = Pattern_ignored.sub('', s)
    words = s.split()
    for i, w in enumerate(words):
        w = w.rstrip(",.:'")

        if Pattern_number.fullmatch(w):
            pass
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
        w = w.rstrip(",.:")

        if not Pattern_number.fullmatch(w):
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


def token2tensor(token: int, batch: int, /):
    return tensor([[token] for _ in range(batch)], device=Device)


@final
class Language:
    __slots__ = 'word_counter', 'word2index', 'index2word',

    def __init__(self, word_counter: Counter[str] = None, /):
        self.word_counter: Counter[str] = word_counter if isinstance(word_counter, Counter) else Counter()
        self._reindex()

    def _reindex(self, /):
        self.index2word: list[str] = Special + sorted(self.word_counter)
        self.word2index: dict[str, int] = {w: i for i, w in enumerate(self.index2word)}

    def __getnewargs__(self, /):
        return self.word_counter,

    @property
    def words_n(self, /):
        return len(self.word2index)

    def add_sentences(self, sentences: list[Sentence], /):
        for sentence in sentences:
            for word in sentence.split():
                self.word_counter[word] += 1

        self._reindex()

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

    def sentence2tokens(self, sentence: Sentence):
        return [self.word2index.get(w, Token_NIL) for w in sentence.split()]


@final
class LanguageData:
    __slots__ = 'data', 'lang', 'max_length',

    def __init__(self, data: list[Sentence], lang: Language, /):
        self.data = data
        self.lang = lang
        self.max_length = max(s.count(' ') for s in data) + 1

        lang.add_sentences(data)

    def __len__(self, /):
        return len(self.data)

    def __iter__(self, /):
        return iter(self.data)

    def __getitem__(self, index: int, /):
        tokens = self.lang.sentence2tokens(self.data[index])
        return tensor(tokens + [PAD] * self.max_length, device=Device)

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

    def __getitem__(self, index: int, /):
        return self.ru[index], self.en[index]

    def get(self, index: int, /):
        return self.ru.get(index), self.en.get(index)


@final
class ParaCrawl(RUENDataset):
    def __init__(self, en_ru_file: str, ru_lang: Language, en_lang: Language, /,
                 data_slice: slice = slice(None)):
        with open(en_ru_file, 'r') as f:
            lines = f.readlines()

        en_list = []
        ru_list = []
        for line in lines[data_slice]:
            en, ru = line.split('\t')
            en_list.append(en)
            ru_list.append(ru)

        super().__init__(ru_list, en_list, ru_lang, en_lang)


@final
class Yandex(RUENDataset):
    def __init__(self, ru_file: str, en_file: str, ru_lang: Language, en_lang: Language, /,
                 data_slice: slice = slice(None)):
        with open(ru_file) as f:
            ru_list = f.readlines()[data_slice]

        with open(en_file) as f:
            en_list = f.readlines()[data_slice]

        super().__init__(ru_list, en_list, ru_lang, en_lang)
