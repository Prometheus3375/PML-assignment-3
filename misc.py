from collections.abc import Callable
from timeit import default_timer


def time2m_s(seconds: float) -> tuple[int, int]:
    seconds = round(seconds)
    minutes = seconds // 60
    seconds -= minutes * 60
    return minutes, seconds


class Timer:
    __slots__ = '_start', 'message'

    def __init__(self, message: str = '\nTime elapsed: %dm %ds', /):
        self._start = 0.
        self.message = message

    def __enter__(self, /):
        self._start = default_timer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb, /):
        print(self.message % time2m_s(default_timer() - self._start))


def time(func: Callable):
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

        m, s = time2m_s(default_timer() - start)
        print(f'\nTime elapsed: {m}m {s}s')

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
