from typing import NamedTuple

from torch.utils.data import Dataset


class ENRU(NamedTuple):
    en: str
    ru: str

    @staticmethod
    def ruen(ru: str, en: str, /):
        return ENRU(en, ru)


class ParaCrawl(Dataset):
    def __init__(self, en_ru_file: str, /):
        with open(en_ru_file, 'r') as f:
            data = [ENRU(*line.strip().split('\t')) for line in f]

        self._data = data

    def __len__(self, /):
        return len(self._data)

    def __iter__(self, /):
        return iter(self._data)

    def __reversed__(self, /):
        return reversed(self._data)

    def __getitem__(self, index: int, /) -> ENRU:
        return self._data[index]
