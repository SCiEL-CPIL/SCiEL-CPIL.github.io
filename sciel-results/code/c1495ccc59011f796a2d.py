"""Spectral-domain B2[2] construction: FFT convolution + greedy + local search, deadline-guarded."""
import time
import numpy as np
from scipy.fft import rfft, irfft

N = 1000
DEADLINE = time.time() + 125.0  # safety margin under the 150s kill
MAX_REP = 2

# Precompute the size-2048 real FFT of the unit basis vectors.
# idx = 2*x  (x in 1..1000) so that convolution index 2*(a+b) = 2a+2b.
_FFT_SIZE = 2048
_UNIT_FFT = np.zeros(_FFT_SIZE // 2 + 1, dtype=np.complex128)
for x in range(1, 1001):
    _UNIT_FFT[x * 2] = 1.0


def _counts_from_indices(indices):
    """Return the exact integer representation-count vector via FFT convolution.

    counts[s] = #{pairs a<=b in indices with a+b == s}, for s in 2..2000.
    """
    if len(indices) == 0:
        return np.zeros(2001, dtype=np.int64)
    idx = np.array(indices, dtype=np.int64) * 2
    f = np.zeros(_FFT_SIZE // 2 + 1, dtype=np.complex128)
    np.add.at(f, idx, 1.0)
    conv = np.real(irfft(f * _UNIT_FFT, n=_FFT_SIZE))
    conv = np.rint(conv)
    return conv[2:2001].astype(np.int64)


def _is_valid(counts):
    return bool(counts.max() <= MAX_REP)


def _greedy(order, mode):
    """Greedy construction over candidate order, keeping a running count dict."""
    chosen = []
    counts = np.zeros(2001, dtype=np.int64)
    for x in order:
        # sums involving x: x+y for y in chosen (y<x) plus 2x
        # representation delta for sum s: 1 for each y in chosen with x+y==s, plus 1 for 2x
        ok = True
        # 2x
        s2 = 2 * x
        if counts[s2] >= MAX_REP:
            ok = False
        if ok:
            for y in chosen:
                s = x + y
                if counts[s] >= MAX_REP:
                    ok = False
                    break
        if not ok:
            continue
        # commit
        counts[s2] += 1
        for y in chosen:
            counts[x + y] += 1
        chosen.append(x)
        if time.time() > DEADLINE - 5:
            break
    return chosen


def _local_search(chosen, budget):
    """Improve via add and swap moves using FFT count recomputation (O(N log N) each)."""
    chosen = list(chosen)
    counts = _counts_from_indices(chosen)
    n = len(chosen)
    in_set = np.zeros(N + 1, dtype=bool)
    for x in chosen:
        in_set[x] = True
    cand = [x for x in range(1, 1001) if not in_set[x]]
    improved = True
    while improved and time.time() < budget:
        improved = False
        # try add moves
        for x in cand:
            s2 = 2 * x
            if counts[s2] >= MAX_REP:
                continue
            ok = True
            for y in chosen:
                if counts[x + y] >= MAX_REP:
                    ok = False
                    break
            if ok:
                new_counts = counts.copy()
                new_counts[s2] += 1
                for y in chosen:
                    new_counts[x + y] += 1
                if _is_valid(new_counts):
                    counts = new_counts
                    chosen.append(x)
                    in_set[x] = True
                    cand.remove(x)
                    improved = True
                    break
        if time.time() > budget:
            break
        # try swap moves: remove a, add b (b not in set)
        for a in list(chosen):
            # counts after removing a
            base = counts.copy()
            base[2 * a] -= 1
            for y in chosen:
                if y != a:
                    base[a + y] -= 1
            for b in cand:
                if b == a:
                    continue
                s2 = 2 * b
                if base[s2] >= MAX_REP:
                    continue
                ok = True
                for y in chosen:
                    if y == a:
                        continue
                    if base[b + y] >= MAX_REP:
                        ok = False
                        break
                if ok:
                    new_counts = base.copy()
                    new_counts[s2] += 1
                    for y in chosen:
                        if y == a:
                            continue
                        new_counts[b + y] += 1
                    if _is_valid(new_counts):
                        counts = new_counts
                        chosen.remove(a)
                        chosen.append(b)
                        in_set[a] = False
                        in_set[b] = True
                        cand.remove(b)
                        cand.append(a)
                        improved = True
                        break
            if improved:
                break
    return chosen


def construct():
    global DEADLINE
    DEADLINE = time.time() + 125.0
    best = []

    # Several greedy passes with different candidate orderings to explore the landscape.
    orders = [
        list(range(1, 1001)),
        list(range(1000, 0, -1)),
        # spread-out: even then odd
        [x for x in range(1, 1001, 2)] + [x for x in range(2, 1001, 2)],
        # multiples of 3 first then rest
        [x for x in range(3, 1001, 3)] + [x for x in range(1, 1001) if x % 3 != 0],
        # random shuffles
    ]
    import random
    rng = random.Random(12345)
    for _ in range(4):
        o = list(range(1, 1001))
        rng.shuffle(o)
        orders.append(o)

    for i, order in enumerate(orders):
        if time.time() > DEADLINE - 8:
            break
        g = _greedy(order, "spread")
        if len(g) > len(best):
            best = g

    # Local-search polish on the best greedy result (and a couple of alternatives).
    candidates_to_polish = [best]
    # also polish the first greedy (ascending) if different
    asc = _greedy(list(range(1, 1001)), "spread") if time.time() < DEADLINE - 10 else []
    if len(asc) > len(best) or (len(asc) == len(best) and asc != best):
        candidates_to_polish.append(asc)

    for c in candidates_to_polish:
        if time.time() > DEADLINE - 6:
            break
        polished = _local_search(c, budget=DEADLINE - 6)
        if len(polished) > len(best):
            best = polished

    # Final validity check
    if best:
        cnt = _counts_from_indices(best)
        if not _is_valid(cnt):
            # fall back to a safe ascending greedy
            best = _greedy(list(range(1, 1001)), "spread")
    if not best:
        best = _greedy(list(range(1, 1001)), "spread")

    return list(best)


if __name__ == "__main__":
    res = construct()
    print("size:", len(res))
