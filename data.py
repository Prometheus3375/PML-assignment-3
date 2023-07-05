import re
from collections import Counter
from collections.abc import Collection
from math import ceil
from typing import final

from torch import tensor
from torch.utils.data import Dataset

Sentence = list[str]

Pattern_punctuation = re.compile(r'([.!?])')
Pattern_ignored = re.compile(r'[^\w.!?]+')


def preprocess_en(s: str, /) -> Sentence:
    s = s.lower()
    s = Pattern_punctuation.sub(r' \1', s)
    s = Pattern_ignored.sub(' ', s)
    return s.strip().split()


def preprocess_ru(s: str, /) -> Sentence:
    s = s.lower()
    s = Pattern_punctuation.sub(r' \1', s)
    s = Pattern_ignored.sub(' ', s)
    return s.strip().split()


Token_sos = 0
Token_eos = 1
Token_nil = 2
SOS = '^'
EOS = '$'
NIL = '*'


@final
class Language:
    __slots__ = 'word_counter', 'word2index', 'index2word',

    def __init__(self, word_counter: Counter[str] = None, /):
        self.word_counter: Counter[str] = word_counter if isinstance(word_counter, Counter) else Counter()
        self._reindex()

    def _reindex(self, /):
        self.index2word: list[str] = [SOS, EOS, NIL] + sorted(self.word_counter)
        self.word2index: dict[str, int] = {i: w for i, w in enumerate(self.index2word)}

    def __getnewargs__(self, /):
        return self.word_counter,

    @property
    def words_n(self, /):
        return len(self.word2index)

    def add_sentences(self, sentences: list[Sentence], /):
        for sentence in sentences:
            for word in sentence:
                self.word_counter[word] += 1

        self._reindex()

    def drop_words(self, percentage: float, /) -> set[str]:
        if percentage > 0:
            infrequent_words_n = ceil(self.words_n * percentage)
            infrequent_words = set(sorted(self.word_counter, key=lambda w: self.word_counter[w])[:infrequent_words_n])

            for word in infrequent_words:
                del self.word_counter[word]

            self._reindex()

            return infrequent_words

        if percentage >= 1:
            raise ValueError(f'cannot drop all words')

        return set()


class RUENDataset(Dataset):
    def __init__(self, ru: Collection[str], en: Collection[str], ru_lang: Language, en_lang: Language, /):
        if len(ru) != len(en):
            raise ValueError(f'different number of sentences: ru ({len(ru):,}) and en ({len(en):,})')

        if ru_lang is en_lang:
            raise ValueError(f'same language object is used for RU and EN')

        self._ru_lang = ru_lang
        self._ru = [preprocess_ru(s) for s in ru]
        ru_lang.add_sentences(self._ru)

        self._en_lang = en_lang
        self._en = [preprocess_en(s) for s in en]
        en_lang.add_sentences(self._en)

    def __len__(self, /):
        return len(self._ru)

    def __iter__(self, /):
        return zip(self._ru, self._en)

    def __getitem__(self, index: int, /):
        ru = [self._ru_lang.word2index[w] for w in self._ru[index]]
        en = [self._en_lang.word2index[w] for w in self._en[index]]
        return tensor(ru), tensor(en)

    def get(self, index: int, /):
        return self._ru[index], self._en[index]

    def drop_ru_words(self, percentage: float, /):
        dropped = self._ru_lang.drop_words(percentage)
        for sentence in self._ru:
            for i, word in enumerate(sentence):
                if word in dropped:
                    sentence[i] = NIL


@final
class ParaCrawl(RUENDataset):
    def __init__(self, en_ru_file: str, ru_lang: Language, en_lang: Language, /):
        with open(en_ru_file, 'r') as f:
            data = (line.strip().split('\t') for line in f)
            data = zip(*data)

        en = next(data)
        ru = next(data)
        super().__init__(ru, en, ru_lang, en_lang)
