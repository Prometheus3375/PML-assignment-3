from collections.abc import Callable


def time(func: Callable):
    from timeit import default_timer
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

        end = default_timer()
        seconds = round(end - start)
        minutes = seconds // 60
        seconds -= minutes * 60
        print(f'\nTime elapsed: {minutes}m {seconds}s')

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
