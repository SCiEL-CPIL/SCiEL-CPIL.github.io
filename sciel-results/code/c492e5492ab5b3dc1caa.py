"""Rewritten program: branch-and-bound on the 'a | b*c' 3-uniform conflict."""
# EVOLVE-BLOCK-START
import time
import random
from math import isqrt

_N = 256
_CAND = list(range(2, _N + 1))
_IDX = {v: i for i, v in enumerate(_CAND)}
_M = len(_CAND)

# Precompute, for each candidate a, the set of candidate indices j,k (j<k)
# such that a divides CAND[j]*CAND[k].
_CONFLICTS = []
for a in _CAND:
    s = set()
    for j in range(_M):
        vj = _CAND[j]
        for k in range(j + 1, _M):
            if (vj * _CAND[k]) % a == 0:
                s.add((j, k))
    _CONFLICTS.append(s)

# Upper bound: max number of pairs a candidate can "conflict" with.
_MAXDEG = max(len(s) for s in _CONFLICTS)

_DEADLINE = None
_TIMEOUT = 75.0  # safety margin under the 90s wall-clock kill


def _valid(set_idx):
    """Check that for every distinct a,b,c in set_idx, a does not divide b*c."""
    L = list(set_idx)
    n = len(L)
    for i in range(n):
        a = _CAND[L[i]]
        cs = _CONFLICTS[L[i]]
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                if (L[j], L[k]) in cs:
                    return False
    return True


def _branch_and_bound(order, budget):
    """Branch-and-bound maximising a valid subset of candidates (by index)."""
    best = []
    best_size = 0

    def _bound(sel_idx, remaining):
        # sel_idx: sorted list of chosen indices
        # remaining: sorted list of candidate indices still available
        # Upper bound = len(sel) + len(remaining)
        return len(sel_idx) + len(remaining)

    def _rec(sel_idx, remaining):
        nonlocal best, best_size
        if time.time() > budget:
            return
        if not remaining:
            if len(sel_idx) > best_size:
                best_size = len(sel_idx)
                best = list(sel_idx)
            return
        # Simple bound
        if _bound(sel_idx, remaining) <= best_size:
            return

        # Branch on the first remaining candidate
        first = remaining[0]
        rest = remaining[1:]

        # Option 1: include first
        # Check that first does not conflict with any pair in sel_idx
        ok = True
        cs = _CONFLICTS[first]
        for i in range(len(sel_idx)):
            for j in range(i + 1, len(sel_idx)):
                if (sel_idx[i], sel_idx[j]) in cs:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            sel_idx.append(first)
            _rec(sel_idx, rest)
            sel_idx.pop()

        # Option 2: exclude first
        _rec(sel_idx, rest)

    _rec([], list(order))
    return best


def construct():
    global _DEADLINE
    t0 = time.time()
    _DEADLINE = t0 + _TIMEOUT

    # Heuristic order: prefer larger numbers (they conflict less)
    order_desc = list(range(_M - 1, -1, -1))

    # Greedy seed to get a good initial lower bound
    greedy_sel = []
    for idx in order_desc:
        ok = True
        cs = _CONFLICTS[idx]
        for i in range(len(greedy_sel)):
            for j in range(i + 1, len(greedy_sel)):
                if (greedy_sel[i], greedy_sel[j]) in cs:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            greedy_sel.append(idx)

    best_size = len(greedy_sel)
    best = list(greedy_sel)

    # Branch and bound with a time budget
    budget = t0 + _TIMEOUT - 2.0  # leave 2s margin
    res = _branch_and_bound(order_desc, budget)
    if len(res) > best_size:
        best_size = len(res)
        best = res

    # If we still have time, try a few random-order B&B runs
    if time.time() < t0 + _TIMEOUT - 5.0:
        rng = random.Random(12345)
        for trial in range(20):
            if time.time() > t0 + _TIMEOUT - 5.0:
                break
            shuffled = order_desc[:]
            rng.shuffle(shuffled)
            budget2 = t0 + _TIMEOUT - 3.0
            res2 = _branch_and_bound(shuffled, budget2)
            if len(res2) > best_size:
                best_size = len(res2)
                best = res2

    # Final safety: ensure validity
    if not _valid(set(best)):
        # Fallback to the prime-only baseline
        best = [i for i, a in enumerate(_CAND)
                if all(a % d for d in range(2, isqrt(a) + 1))]

    result = [_CAND[i] for i in best]
    # Ensure distinct ints in 2..256, return as list of Python ints
    result = sorted(set(int(x) for x in result))
    return result
# EVOLVE-BLOCK-END
