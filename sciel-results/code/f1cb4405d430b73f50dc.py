"""Kissing number in dimension 11: maximize # integer vectors with
min pairwise distance >= max norm. Validity guaranteed by greedy construction."""
import time
import itertools
import numpy as np

D = 11
_DEADLINE = 150.0
_MARGIN = 6.0


def _candidate_pool():
    pool = []
    # weight 2
    for i, j in itertools.combinations(range(D), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0] * D
                v[i], v[j] = si, sj
                pool.append(tuple(v))
    # weight 3
    for i, j, k in itertools.combinations(range(D), 3):
        for si in (1, -1):
            for sj in (1, -1):
                for sk in (1, -1):
                    v = [0] * D
                    v[i], v[j], v[k] = si, sj, sk
                    pool.append(tuple(v))
    return pool


def construct():
    t0 = time.time()
    pool = _candidate_pool()
    chosen = []
    norms2 = []
    for v in pool:
        if time.time() - t0 > _DEADLINE - _MARGIN:
            break
        vec = np.array(v, dtype=np.int64)
        n2 = int(np.dot(vec, vec))
        if chosen:
            dots = np.array(chosen, dtype=np.int64) @ vec
            d2 = n2 + np.array(norms2) - 2 * dots
            if d2.min() < n2:
                continue
        chosen.append(vec)
        norms2.append(n2)
    if chosen:
        return np.array(chosen, dtype=np.int64)
    return np.zeros((0, D), dtype=np.int64)
