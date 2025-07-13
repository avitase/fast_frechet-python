from functools import partial, reduce
from itertools import accumulate

import numpy as np


def frechet_combine(a, b):
    v1, d1 = a
    v2, d2 = b
    v, d = min(v1, v2), max(d1, d2)
    return d if d2 > v1 else v, v if d1 > v2 else d


def frechet_next(v, d):
    v[1:] = np.minimum(v[:-1], v[1:])
    v = np.maximum(v, d)

    d[0] = v[0]
    return [d for _, d in accumulate(zip(v, d), frechet_combine)]


def frechet_distance(p, q, metric):
    d = map(partial(metric, q), p)
    init = np.maximum.accumulate(next(d))
    return reduce(frechet_next, d, init)[-1]
