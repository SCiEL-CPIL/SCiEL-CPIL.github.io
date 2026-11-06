"""Find the maximum-size dissociated subset of [1..22].

A subset S is dissociated (reciprocal) iff every subset of S has a distinct
sum of reciprocals: the map A -> sum_{a in A} 1/a is injective over all 2^|S|
sub-sums (including the empty sum = 0).

Strategy:
  - Represent reciprocal sums as integers scaled by LCM(1..22) for exact
    arithmetic without Fraction overhead.
  - Branch-and-bound over candidate elements, trying multiple orderings to
    escape local optima.
  - Exact collision test: v is addable iff (v - s) not in current sums for all s.
  - Global deadline guard with safety margin.
"""
import time
from math import gcd
from functools import reduce

_T0 = time.time()
_BUDGET = 75.0  # wall-clock guard (evaluator kills at 90s; keep 15s margin)
_N = 22

# LCM(1..22) for exact integer representation of reciprocal sums
def _lcm(a, b):
    return a // gcd(a, b) * b

_LCM = reduce(_lcm, range(1, _N + 1))
# Reciprocal of i as integer: LCM // i
_REC = [0] + [_LCM // i for i in range(1, _N + 1)]


def _is_dissociated(elements):
    """Independent verification: all 2^k reciprocal sub-sums distinct."""
    sums = {0}
    for e in elements:
        v = _REC[e]
        shifted = {s + v for s in sums}
        if shifted & sums:
            return False
        sums |= shifted
    return True


def _search_order(order, best_ref, deadline):
    """Branch-and-bound with a fixed ordering. best_ref = [card, elements]."""
    n = len(order)
    cur_sums = {0}
    cur_set = []

    def dfs(idx):
        if time.time() > deadline:
            return
        # Bound: cannot beat best
        if len(cur_set) + (n - idx) <= best_ref[0]:
            return
        if idx == n:
            if len(cur_set) > best_ref[0]:
                best_ref[0] = len(cur_set)
                best_ref[1] = list(cur_set)
            return

        v = _REC[order[idx]]
        # Try include first (stronger bound pruning)
        # Collision test: v addable iff (v - s) not in cur_sums for all s
        can = True
        for s in cur_sums:
            if (v - s) in cur_sums:
                can = False
                break
        if can:
            new_sums = cur_sums | {s + v for s in cur_sums}
            cur_set.append(order[idx])
            # Swap cur_sums in
            dfs_inner(idx + 1, new_sums)
            cur_set.pop()

        # Exclude
        dfs_inner(idx + 1, cur_sums)

    def dfs_inner(idx, sums):
        if time.time() > deadline:
            return
        if len(cur_set) + (n - idx) <= best_ref[0]:
            return
        if idx == n:
            if len(cur_set) > best_ref[0]:
                best_ref[0] = len(cur_set)
                best_ref[1] = list(cur_set)
            return
        v = _REC[order[idx]]
        can = True
        for s in sums:
            if (v - s) in sums:
                can = False
                break
        if can:
            new_sums = sums | {s + v for s in sums}
            cur_set.append(order[idx])
            dfs_inner(idx + 1, new_sums)
            cur_set.pop()
        dfs_inner(idx + 1, sums)

    dfs_inner(0, {0})


def _verified_baseline():
    seed = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    return seed if _is_dissociated(seed) else []


def construct():
    """Return a nonempty subset of [1..22] maximising dissociated cardinality."""
    fallback = _verified_baseline()
    best = [len(fallback), list(fallback)]

    # Try multiple orderings to explore different search trees
    # Ordering 1: descending (largest reciprocals first) - current approach
    # Ordering 2: ascending
    # Ordering 3: specific heuristic orderings
    deadline = _T0 + _BUDGET

    orderings = [
        list(range(_N, 0, -1)),  # descending: 22, 21, ..., 1
        list(range(1, _N + 1)),  # ascending: 1, 2, ..., 22
        # Alternate: odd first then even, descending within
        [i for i in range(_N, 0, -2)] + [i for i in range(_N - 1, 0, -2)],
        # Primorial-ish: primes first descending, then composites
        [19, 17, 13, 11, 7, 5, 3, 2, 22, 21, 20, 18, 16, 15, 14, 12, 10, 9, 8, 6, 4, 1],
        # Reverse of the above
        [1, 4, 6, 8, 9, 10, 12, 14, 15, 16, 18, 20, 21, 22, 2, 3, 5, 7, 11, 13, 17, 19],
    ]

    for order in orderings:
        if time.time() > deadline:
            break
        _search_order(order, best, deadline)

    result = best[1]
    candidates = [c for c in (result, fallback) if c and _is_dissociated(c)]
    if not candidates:
        candidates = [[1]]
    candidates.sort(key=len, reverse=True)
    return candidates[0]
