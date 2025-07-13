import timeit
from functools import partial

import numpy as np

from fast_frechet import (
    accumulate,
    branchless,
    compiled,
    linear_memory,
    no_recursion,
    reduce_accumulate,
    reduce_accumulate2,
    vectorized,
)


def metric(p, q):
    dx = p[..., 0] - q[..., 0]
    dy = p[..., 1] - q[..., 1]
    return np.hypot(dx, dy)


def generate_trajectory(n, *, rng):
    xy0 = rng.integers(-2, 2, size=(1, 2), endpoint=False).astype(np.float64)
    dxy = rng.integers(-1, 1, size=(n, 2), endpoint=False).astype(np.float64)
    return xy0 + np.cumsum(dxy, axis=0)


def benchmark(f, *, n, rng):
    p = generate_trajectory(n, rng=rng)
    q = generate_trajectory(n, rng=rng)
    return min(timeit.repeat(lambda: f(p, q), repeat=3, number=1)) * 1_000


def main(*, n=1024, seed=42):
    print(f"Length of trajectory = {n}")
    print("")

    for v in [
        no_recursion,
        vectorized,
        branchless,
        linear_memory,
        accumulate,
        reduce_accumulate,
        reduce_accumulate2,
        compiled,
    ]:
        f = partial(v.frechet_distance, metric=metric)
        t = benchmark(f, n=n, rng=np.random.default_rng(seed))
        print(f"{v.__name__.split('.')[-1]:>20}: {t:>4.0f} ms")


if __name__ == "__main__":
    main()
