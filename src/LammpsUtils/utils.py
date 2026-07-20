import random
from pathlib import Path
import numpy as np
from numpy.polynomial import Polynomial
from scipy.spatial import cKDTree

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

def product(x: list[int|float]):
    """Compute the product of items in a list, similar to the built-in sum()."""
    prod = 1
    for v in x:
        prod *= v
    return prod

def create_seeds(num_seeds: int = None, bounds=(0, 1000000)):
    """Create a seed or a list of unique seeds with an integer value in the provided bounds."""
    random.seed()
    if num_seeds is None:
        return random.randint(bounds[0], bounds[1])
     
    seeds = {}
    while len(seeds) < num_seeds:
        seeds.update({random.randint(bounds[0], bounds[1]): None})
    return list(seeds.keys())

def random_range(start, stop, step=1, seed=None):
    """Creates a randomized range of integers."""
    rng = np.random.default_rng(seed=seed)
    values = np.arange(start, stop, step)
    return rng.permutation(values).tolist()

def warren_cowley(num_neighbors: int, shell_radii: list[float], positions: np.ndarray, types: np.ndarray, boxlo:np.ndarray, boxsize: np.ndarray):
    """Compute the Warren-Cowley parameters of a configuration given the simulation box size, atomic positions, and types."""
    # list like [1, 2, 1, 1] -> list like [1, 2]
    unique_types = sorted(list(set(types))) 
    num_unique_types = len(unique_types)

    # move box back to origin and correct positions
    positions = positions - boxlo
    
    # coordinates are not required to be within the box, so wrap any that are outside the box
    positions = positions.round(decimals=4)
    unw_num_imgs = np.floor_divide(positions, boxsize)
    positions = positions - unw_num_imgs*boxsize

    # k-d trees have O(log n) speed
    position_tree = cKDTree(positions, boxsize=boxsize)

    # vector of square matrices (each is a shell) where rows are reference atoms types and columns are number of neighbors of each type
    num_shells = len(shell_radii)-1
    neighbors = np.zeros((num_shells, num_unique_types, num_unique_types))
    wc = np.zeros((num_shells, num_unique_types, num_unique_types))

    # compute composition using types array
    composition = {int(t): np.sum(np.where(types==t, 0, 1))/len(types) for t in unique_types}

    # get number of neighbors of each type for each atom (using mininum image convention)
    for pos in positions:
        neigh_dist, neigh_idcs = position_tree.query(pos, k=num_neighbors+1)

        # get central atom type and remove it from lists
        ref_type = types[neigh_idcs[0]]
        neigh_dist, neigh_idcs = neigh_dist[1:],  neigh_idcs[1:]

        for shi in range(num_shells):
            # atom position indices within shell
            shell_mask = (neigh_dist > shell_radii[shi]) & (neigh_dist < shell_radii[shi+1])
            shell_idcs = neigh_idcs[shell_mask]
            
            for ni in shell_idcs:
                neigh_type = types[ni]
                neighbors[shi, ref_type-1, neigh_type-1] += 1

    # compute all possible paramaters as an NxN matrix where N is the number types following the same convention as neighbors matrices
    for shi in range(num_shells): 
        for to in unique_types:
            for ti in unique_types:
                wc[shi, to-1, ti-1] = 1 - (neighbors[shi, to-1, ti-1] / np.sum(neighbors[shi, to-1, :])) / composition[to]

    return wc

