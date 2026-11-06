# EVOLVE-BLOCK-START
import time
from math import isqrt
import numpy as np

def _gen_powerful(limit):
    max_b = int(round(limit ** (1.0 / 3.0))) + 2
    is_sqfree = np.ones(max_b + 1, dtype=bool)
    is_sqfree[0] = False
    for p in range(2, int(isqrt(max_b)) + 1):
        if is_sqfree[p]:
            p2 = p * p
            for k in range(p2, max_b + 1, p2):
                is_sqfree[k] = False
    sqfree_b = [b for b in range(1, max_b + 1) if is_sqfree[b]]
    out = set()
    for b in sqfree_b:
        b3 = b ** 3
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            out.add(a * a * b3)
    return sorted(out)

def construct():
    start = time.time()
    deadline = start + 135

    L = 100_000_000
    pow = _gen_powerful(L)
    if time.time() > deadline:
        L = 20000
        pow = _gen_powerful(L)
    pow_arr = np.array(pow, dtype=np.int64)
    n = len(pow_arr)
    if n < 2 or time.time() > deadline:
        return {'N': 2, 'pairs': [[1, 1]]}

    max_sum = 2 * L
    hist = np.zeros(max_sum + 1, dtype=np.int64)

    chunk = 10000
    for i0 in range(0, n, chunk):
        if time.time() > deadline:
            break
        a_chunk = pow_arr[i0:i0 + chunk]
        # collect all sums for this chunk into a list of arrays, then one bincount
        parts = []
        for a in a_chunk:
            b_start = np.searchsorted(pow_arr, a, side='left')
            if b_start < n:
                parts.append(a + pow_arr[b_start:])
        if parts:
            all_sums = np.concatenate(parts)
            hist += np.bincount(all_sums, minlength=max_sum + 1)

    best_N = int(np.argmax(hist))
    best_count = int(hist[best_N])
    if best_count < 1:
        return {'N': 2, 'pairs': [[1, 1]]}

    members = set(pow)
    pairs = []
    for a in pow:
        if 2 * a > best_N:
            break
        b = best_N - a
        if b in members:
            pairs.append([a, b])
    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
