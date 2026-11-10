# EVOLVE-BLOCK-START
import numpy as np
import itertools
import time


def no_5_on_a_sphere8() -> np.ndarray:
    """
    The "no 5 on a sphere" problem. Select as many points as possible from the
    n x n x n integer grid {0,...,n-1}^3 such that no five of the selected points
    are co-spherical (lie on a common sphere, which also forbids five co-planar
    points).

    Strategy: incremental greedy with smarter ordering and multiple restarts.
    """
    n = 8
    num_restarts = 30

    grid = np.array(list(itertools.product(range(n), range(n), range(n))), dtype=np.int64)

    def can_add(arr: np.ndarray, p: np.ndarray) -> bool:
        m = len(arr)
        if m < 4:
            return True
        w = np.empty((m, 4), dtype=np.float64)
        w[:, 0:3] = arr - p
        w[:, 3] = (arr ** 2).sum(axis=1) - (p ** 2).sum()
        combos = list(itertools.combinations(range(m), 4))
        for combo in combos:
            sub = w[list(combo)]
            if abs(np.linalg.det(sub)) < 0.5:
                return False
        return True

    deadline = time.time() + 580  # 580s to be safe

    np.random.seed(0)
    best = None
    for _ in range(num_restarts):
        if time.time() > deadline - 10:
            break
        order = grid[np.random.permutation(len(grid))]
        arr = np.empty((0, 3), dtype=np.int64)
        for p in order:
            if can_add(arr, p):
                arr = np.vstack([arr, p]) if len(arr) else p[None, :]
        if best is None or len(arr) > len(best):
            best = arr

    return best.astype(np.int64)


# EVOLVE-BLOCK-END
