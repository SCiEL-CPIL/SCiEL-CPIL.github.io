import time
import random
import numpy as np

_N = 256
_DEADLINE = 140.0  # explicit safety margin before 150s kill

# ---- Precompute all 4-term APs (0-indexed) ----
_APS = np.array(
    [
        (a - 1, a - 1 + d, a - 1 + 2 * d, a - 1 + 3 * d)
        for d in range(1, 86)
        for a in range(1, _N - 3 * d + 1)
    ],
    dtype=np.int32,
)
_NAPS = _APS.shape[0]
# per-AP sign vectors so that mono = (sum of signs * colors) == 4
_SIGNS = np.array([1, -1, 1, -1], dtype=np.int8)
# column-major so that C @ S is a contiguous slice (fast)
_S = (_SIGNS[:, None] * _APS.T).T  # shape (4, NAPS)
_S = np.ascontiguousarray(_S, dtype=np.int8)


def _count(colors):
    """Exact number of monochromatic 4-APs (vectorized)."""
    return int((np.abs(colors[_APS]) @ _S == 4).sum())


def _deltas(colors):
    """Delta (change in count) if we flip each of the 256 positions.
    Positive delta = flipping increases the count (bad)."""
    C = colors[_APS]  # (NAPS, 4)
    base = np.abs(C) @ _S  # (NAPS,)
    mono_now = base == 4
    mono_after = (-base) == 4  # flipping a position negates its signed contribution
    return mono_after.astype(np.int32) - mono_now.astype(np.int32)  # (NAPS,)


def _batch_deltas(colors):
    """Vectorized delta for flipping every position simultaneously (all 256)."""
    per_ap = _deltas(colors)  # (NAPS,)
    return per_ap[_APS].sum(axis=1)  # (256,)


def _thue_morse():
    """Thue-Morse initial coloring: c[i] = popcount(i) mod 2 (low discrepancy)."""
    return np.array([bin(i).count("1") & 1 for i in range(_N)], dtype=np.int8)


def _local_search(colors, count, deadline):
    """Batch local search: each step flips the single best-improving position.
    Repeats until no improving flip exists or the deadline is hit."""
    improved = True
    while improved and time.time() < deadline - 1:
        improved = False
        d = _batch_deltas(colors)  # delta if flipping each position
        best = int(d.argmin())
        if d[best] < 0:
            colors[best] ^= 1
            count += d[best]
            improved = True
    return colors, count


def _sa(colors, count, deadline, T0, Tmin, cool):
    """Simulated annealing with vectorized counting and fast local-search polish."""
    T = T0
    while time.time() < deadline - 1:
        pos = random.randrange(_N)
        colors[pos] ^= 1
        c2 = _count(colors)
        delta = c2 - count
        if delta <= 0 or random.random() < np.exp(-delta / max(T, 1e-6)):
            count = c2
        else:
            colors[pos] ^= 1  # revert
        T = max(T * cool, Tmin)
        if (time.time() < deadline - 1) and (T <= Tmin or random.random() < 0.02):
            colors, count = _local_search(colors, count, deadline)
    return colors, count


def construct():
    start = time.time()
    deadline = start + _DEADLINE
    rng = random.Random(12345)

    best_colors = None
    best_count = 10 ** 9

    def consider(colors, count):
        nonlocal best_colors, best_count
        if count < best_count:
            best_count = count
            best_colors = colors.copy()

    # ---- Phase 1: Thue-Morse start + local search ----
    c = _thue_morse()
    c, n = _local_search(c, _count(c), deadline)
    consider(c, n)

    # ---- Phase 2: simulated annealing from Thue-Morse ----
    c = _thue_morse()
    c, n = _sa(c, _count(c), deadline, T0=3.0, Tmin=0.02, cool=0.999)
    consider(c, n)

    # ---- Phase 3: random restarts + local search + light SA ----
    while time.time() < deadline - 2:
        c = np.array([rng.randrange(2) for _ in range(_N)], dtype=np.int8)
        c, n = _local_search(c, _count(c), deadline)
        consider(c, n)
        if time.time() < deadline - 2:
            c, n = _sa(c, n, deadline, T0=1.5, Tmin=0.02, cool=0.999)
            consider(c, n)

    # ---- Final polish ----
    if best_colors is not None:
        bc, bn = _local_search(best_colors.copy(), best_count, deadline)
        consider(bc, bn)

    return [int(x) for x in best_colors]
