from torch.utils.data import Dataset


class ParaCrawl(Dataset):
    def __init__(self, en_ru_file: str, /):
        with open(en_ru_file, 'r') as f:
            data = [tuple(line.strip().split('\t')) for line in f]
        print(data[0])
        self._data: list[tuple[str, str]] = data

    def __len__(self, /):
        return len(self._data)

    def __iter__(self, /):
        return iter(self._data)

    def __reversed__(self, /):
        return reversed(self._data)

    def __getitem__(self, index: int, /) -> tuple[str, str]:
        return self._data[index]
