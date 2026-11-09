# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def _base_cover(q, n, radius):
    # Direct sum of the ternary tetracode (9 words of length 4, perfect covering radius 1)
    # with every word on the remaining n - 4 coordinates: 9 * 3**(n - 4) words.
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(q) for b in range(q)]
    if n <= 4:
        return [list(h[:n]) for h in tetracode]
    return [h + list(tail) for h in tetracode
            for tail in itertools.product(range(q), repeat=n - 4)]


def _is_cover(codewords, q, n, radius):
    if not codewords:
        return n == 0
    C = np.array(codewords, dtype=np.int32).reshape(-1, n)
    words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    # word i is covered by codeword j iff sum_{k} (w_i,k != c_j,k) <= radius
    diff = (words[:, None, :] != C[None, :, :])
    dist = diff.sum(axis=2)
    return bool(np.all(dist.min(axis=1) <= radius))


def _greedy_cover(q, n, radius, deadline):
    if n == 0:
        return [[]]

    words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int32)
    M = words.shape[0]
    diff = (words[:, None, :] != words[None, :, :])
    dist = diff.sum(axis=2)                      # M x M int32 Hamming distances
    del diff
    ball = dist <= radius                        # M x M bool: ball-membership matrix
    col_sums = ball.sum(axis=1).astype(np.int64)  # size of each ball
    del dist
    del words

    covered = np.zeros(M, dtype=bool)
    code_idx = []
    remaining = M

    while remaining > 0:
        if time.time() > deadline:
            break
        cand = ~covered
        if not cand.any():
            break
        gains = np.where(cand, col_sums, -1)
        j = int(np.argmax(gains))
        if gains[j] <= 0:
            # No candidate ball adds new coverage: pick any uncovered word as a ball center.
            j = int(np.argmax(cand))
        code_idx.append(j)
        newcov = ball[j] & ~covered
        if newcov.any():
            covered |= newcov
            remaining -= int(newcov.sum())
    return [list(b) for b in ball[code_idx, :]]


def construct(q, n, radius):
    start = time.time()
    deadline = start + 105.0          # explicit guard, safety margin vs 120s wall-clock

    if n == 0:
        return [[]]

    # Baseline that is guaranteed valid and always available to return.
    base = _base_cover(q, n, radius)
    base = base[:1000000]             # keep memory bounded for large inputs
    if not _is_cover(base, q, n, radius):
        # Fallback: exhaustive (valid but large); only for pathological small q,n.
        base = [list(w) for w in itertools.product(range(q), repeat=n)]
    best = base

    # Greedy refinement: try to beat the baseline within the time budget.
    try:
        cand = _greedy_cover(q, n, radius, deadline)
        if len(cand) < len(best) and _is_cover(cand, q, n, radius):
            best = cand
    except Exception:
        pass

    if time.time() > deadline:
        return best
    return best
# EVOLVE-BLOCK-END
