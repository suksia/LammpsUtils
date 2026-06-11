from pathlib import Path
import numpy as np
from numpy.polynomial import Polynomial

def strip_split(s: str, sep=None, item_type=None):
    """Strip a string of whitespace and split it apart given a separator character."""
    s = s.strip()
    s = s.split(sep)
    if item_type is int:
        return [int(x) for x in s]
    elif item_type is float:
        return [float(x) for x in s]
    elif item_type is None:
        return s
    else:
        raise ValueError(f'[{item_type}] Invalid item type. Choose None, int, or float')

def tilps(list_vals: list, sep: str = ' '):
    """Inverse of strip(), where a list of strings are glued back together into a single string."""
    s = ''
    for l in list_vals:
        s += str(l) + sep
    return s.strip()

def next_path(path: Path):
    """Return a path name with the next available index appended to it (e.g., 'some_path_050')."""
    i = 0
    while i < 1000:
        new_path = path.parent / (path.name+f'_{i:003}')
        if new_path.exists():
            i += 1
        else:
            return new_path
    raise ValueError(f'[{path}] Study file path index limit reached (1000)')

def unprefix(int_prefix: str) -> int:
    int_prefix = str(int_prefix)
    val, prefix = int(int_prefix[:-1]), int_prefix[-1]
    if prefix == 'k':
        return val*1000
    elif prefix == 'M':
        return val*1000000
    else:
        return int(int_prefix)
    
def linear_fit(x, y):
    fit, fit_data = Polynomial.fit(x, y, 1, full=True)
    intercept, slope = fit.convert().coef

    ym = np.mean(y)
    rss = fit_data[0][0]
    tss = np.sum([(yv-ym)**2 for yv in y])
    r_squared = 1 - rss/tss

    return intercept, slope, r_squared

def sign(x):
    if x >= 0:
        return 1
    else:
        return -1