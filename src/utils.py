import random
from pathlib import Path
import numpy as np
from numpy.polynomial import Polynomial

def strip_split(s: str, sep=None, as_type=str):
    """Strip a string of any whitespace or newline characters, split it apart with a separator character, and convert items to given type."""
    s = s.strip()
    s = s.split(sep)
    try:
        return [as_type(x) for x in s]
    except:
        raise ValueError(f'Could not cast all items in {s} to type {as_type}')

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

def unprefix(prefixed_num, as_type=int):
    """Convert a number written with an S.I. prefix to its actual form (e.g., 173k=173000, 1.5M=1500000)."""
    if type(prefixed_num) == str:
        val, prefix = float(prefixed_num[:-1]), prefixed_num[-1]
    else:
        return as_type(prefixed_num)
    
    if prefix == 'k':
        return as_type(val*1000)
    elif prefix == 'M':
        return as_type(val*1000000)
    else:
        raise ValueError(f'Unrecognized prefix {prefix}')
    
def linear_fit(x, y):
    """Fit a line to a dataset and return its parameters."""
    fit, fit_data = Polynomial.fit(x, y, 1, full=True)
    intercept, slope = fit.convert().coef

    ym = np.mean(y)
    rss = fit_data[0][0]
    tss = np.sum([(yv-ym)**2 for yv in y])
    r_squared = 1 - rss/tss

    return intercept, slope, r_squared

def sign(x):
    """Determine the sign of a number."""
    if x >= 0:
        return 1
    else:
        return -1

def create_seeds(num_seeds: int = None, bounds=(0, 1000000)):
    """Create a seed or a list of unique seeds with an integer value in the provided bounds."""
    random.seed()
    if num_seeds is None:
        return random.randint(bounds[0], bounds[1])
     
    seeds = {}
    while len(seeds) < num_seeds:
        seeds.update({random.randint(bounds[0], bounds[1]): None})
    return list(seeds.keys())