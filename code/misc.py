from collections.abc import Callable
from timeit import default_timer


def seconds2hms(seconds: float, /):
    s = round(seconds)
    h = s // 3600
    s -= h * 3600
    m = s // 60
    s -= m * 60
    return h, m, s


def format_hms(h: int, m: int, s: int, /) -> str:
    if h > 0:
        return f'{h}h {m}m {s}s'

    if m > 0:
        return f'{m}m {s}s'

    return f'{s}s'


class Timer:
    __slots__ = '_start', 'message'

    def __init__(self, message: str = '\nTime elapsed: %s', /):
        self._start = 0.
        self.message = message

    def __enter__(self, /):
        self._start = default_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb, /):
        hms = seconds2hms(default_timer() - self._start)
        print(self.message % format_hms(*hms))


def time(func: Callable, /):
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = default_timer()
        exception = None
        try:
            result = func(*args, **kwargs)
        except KeyboardInterrupt:
            result = None
        except Exception as e:
            result = None
            exception = e

        hms = seconds2hms(default_timer() - start)
        print(f'\nTime elapsed: {format_hms(*hms)}')

        if exception:
            raise exception
        return result

    return wrapper


class Printer:
    __slots__ = '_line_length',

    def __init__(self, /):
        self._line_length = 0

    def __enter__(self, /):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb, /):
        print()

    def print(self, message: str, /):
        pad_size = self._line_length - len(message)
        if pad_size > 0:
            pad = ' ' * pad_size
        else:
            pad = ''
            self._line_length = len(message)
        print(f'\r{message}{pad}', end='')
