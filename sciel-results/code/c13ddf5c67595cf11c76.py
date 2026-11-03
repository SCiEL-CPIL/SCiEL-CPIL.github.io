"""Construct a finite set U of nonnegative integers containing 0 to maximize
the lower bound 1 + log(|U-U|/|U+U|)/log(2*max(U)+1).

Design: parametric two-interval family plus local refinement, with an explicit
deadline guard. Two separated blocks inflate |U-U| relative to |U+U| because
cross-block differences span a wide range while cross-block sums overlap with
intra-block sums, keeping |U+U| comparatively small.
"""
import time
import math
import numpy as np

_DEADLINE = 150.0
_MARGIN = 10.0  # safety margin in seconds


def _score(U):
    """Return the lower-bound score for a set U (list/set of ints, must contain 0)."""
    U = sorted(set(int(x) for x in U))
    if 0 not in U:
        return 0.0
    n = len(U)
    if n < 2:
        return 0.0
    Ua = np.array(U, dtype=np.int64)
    # sumset
    s = np.add.outer(Ua, Ua)
    n_sum = len(np.unique(s))
    # diffset
    d = np.subtract.outer(Ua, Ua)
    n_diff = len(np.unique(d))
    mx = Ua[-1]
    denom = math.log(2 * mx + 1)
    if denom <= 0:
        return 0.0
    if n_sum == 0:
        return 0.0
    ratio = n_diff / n_sum
    if ratio <= 1.0:
        return 1.0
    return 1.0 + math.log(ratio) / denom


def _two_interval(a, b, g):
    """U = {0..a-1} union {g..g+b-1}, ensure 0 in U, g >= a so blocks separated."""
    if a < 1:
        a = 1
    if b < 1:
        b = 1
    if g < a:
        g = a
    U = list(range(0, a)) + list(range(g, g + b))
    return sorted(set(U))


def _local_refine(U, max_rounds=30):
    """Local search: try adding/removing single elements to improve score."""
    U = sorted(set(int(x) for x in U))
    if 0 not in U:
        U.append(0)
    U = sorted(set(U))
    best = _score(U)
    bestU = U[:]
    mx = U[-1]
    for _ in range(max_rounds):
        improved = False
        candidates = set()
        # try adding near existing elements and at the tail
        for x in U:
            for dx in (1, 2, 3, -1, -2):
                y = x + dx
                if y >= 0:
                    candidates.add(y)
        candidates.add(mx + 1)
        candidates.add(mx + 2)
        candidates.add(0)
        # try removing (except 0)
        for y in sorted(candidates):
            if y in bestU:
                cand = [z for z in bestU if z != y]
            else:
                cand = bestU + [y]
            s = _score(cand)
            if s > best + 1e-12:
                best = s
                bestU = sorted(cand)
                improved = True
        if not improved:
            break
    return bestU, best


def construct():
    t0 = time.time()
    bestU = [0, 1, 3]
    bestScore = _score(bestU)

    def remaining():
        return _DEADLINE - _MARGIN - (time.time() - t0)

    # Grid search over two-interval family: sizes a,b and gap g.
    # Keep element count and max bounded so numpy outer products stay fast.
    size_range = list(range(1, 40))
    # For each (a,b), search a range of gaps.
    for a in size_range:
        if remaining() <= 0:
            break
        for b in size_range:
            if remaining() <= 0:
                break
            # gap from a (touching) up to a moderate multiple
            max_gap = max(a, b) * 6
            # sample gaps: small steps near a, then coarser
            gaps = list(range(a, a + max(1, max_gap // 10)))
            # add some larger sampled gaps
            extra = [int(a + max_gap * f) for f in (0.25, 0.5, 0.75, 1.0)]
            for g in extra:
                if g >= a and g not in gaps:
                    gaps.append(g)
            for g in gaps:
                if remaining() <= 0:
                    break
                U = _two_interval(a, b, g)
                s = _score(U)
                if s > bestScore:
                    bestScore = s
                    bestU = U

    # Local refinement of the best found set.
    if remaining() > 5:
        refined, s = _local_refine(bestU, max_rounds=40)
        if s > bestScore:
            bestScore = s
            bestU = refined

    # Ensure 0 present and nonnegative ints.
    bestU = sorted(set(int(x) for x in bestU if x >= 0))
    if 0 not in bestU:
        bestU.insert(0, 0)
    return bestU
