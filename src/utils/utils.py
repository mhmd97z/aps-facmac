
def parse_slice(slice_, end=0):
    start = slice_.start if slice_.start is not None else 0
    step = slice_.step if slice_.step is not None else 1
    if slice_.stop is None:
        stop = end
    elif slice_.stop == -1:
        stop = end - 1
    else:
        stop = slice_.stop
    return range(start, stop, step)