import functools


def to_list(func):
    @functools.wraps(func)
    def to_list_wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return list(result)
    return to_list_wrapper


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip(*args)
