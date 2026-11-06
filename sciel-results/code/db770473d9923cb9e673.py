"""Rewritten program: greedy multi-start + local improvement with deadline guard."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

_N = 192
_MAX_K = 48
_DEADLINE = 78.0  # safety margin under 90s wall-clock

def _is_valid_set(sel):
    """Check whether no element of sel divides the sum of any 3 distinct others."""
    s = sorted(sel)
    m = len(s)
    if m < 4:
        return True
    for i, a in enumerate(s):
        # need 3 distinct others
        others = [s[j] for j in range(m) if j != i]
        # if fewer than 3 others, skip
        if len(others) < 3:
            continue
        # check all triples
        for b, c, d in combinations(others, 3):
            if (b + c + d) % a == 0:
                return False
    return True

def _can_add(sel, x):
    """Check whether adding x to sel keeps the set valid. sel is a set (not containing x)."""
    # After adding x, we must ensure:
    # 1) x does not divide the sum of any 3 distinct elements of sel
    # 2) no element a in sel divides the sum of any 3 distinct elements of (sel ∪ {x})
    #    which includes triples that contain x.
    s = list(sel)
    m = len(s)
    # Condition 1: x | (b+c+d) for any 3 distinct in sel
    if m >= 3:
        for b, c, d in combinations(s, 3):
            if (b + c + d) % x == 0:
                return False
    # Condition 2: for each a in sel, check triples that include x
    #   i.e., a | (x + b + c) for any 2 distinct b,c in sel
    if m >= 2:
        for a in s:
            for b, c in combinations(s, 2):
                if (x + b + c) % a == 0:
                    return False
    return True

def _greedy_build(order, start_set=None):
    """Greedy build: try to add each element in `order` if it fits."""
    sel = set(start_set) if start_set else set()
    for x in order:
        if x in sel:
            continue
        if len(sel) >= _MAX_K:
            break
        if _can_add(sel, x):
            sel.add(x)
    return sel

def _local_improve(sel, deadline):
    """Try to add missing elements and do 1-for-1 or 1-for-2 swaps to grow the set."""
    sel = set(sel)
    universe = set(range(1, _N + 1))
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Pass 1: add any missing element that fits
        for x in sorted(universe - sel):
            if len(sel) >= _MAX_K:
                break
            if _can_add(sel, x):
                sel.add(x)
                improved = True
        # Pass 2: try removing one element to allow adding a different (or more) element
        # We look for a swap: remove a, then greedily add as many as possible; keep if larger.
        for a in list(sel):
            if time.time() >= deadline:
                break
            sel2 = sel - {a}
            added = 0
            # try to add any missing element that fits into sel2
            for x in sorted(universe - sel2):
                if len(sel2) >= _MAX_K:
                    break
                if _can_add(sel2, x):
                    sel2.add(x)
                    added += 1
            if len(sel2) > len(sel):
                sel = sel2
                improved = True
    return sel

def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE
    best = set()

    # Deterministic descending greedy (matches seed behavior, gives a valid baseline)
    order_desc = list(range(_N, 0, -1))
    best = _greedy_build(order_desc)

    # Ascending greedy
    if time.time() < deadline:
        order_asc = list(range(1, _N + 1))
        cand = _greedy_build(order_asc)
        if len(cand) > len(best):
            best = cand

    # Randomized multi-start greedy
    rng = random.Random(12345)
    while time.time() < deadline:
        order = list(range(1, _N + 1))
        rng.shuffle(order)
        cand = _greedy_build(order)
        # local improvement on the candidate
        cand = _local_improve(cand, deadline)
        if len(cand) > len(best):
            best = cand
            if len(best) >= _MAX_K:
                break

    # Final local improvement on the best set found
    if time.time() < deadline:
        best = _local_improve(best, deadline)

    # Ensure validity of the returned set (defensive)
    if not _is_valid_set(list(best)):
        # Fallback to a known-valid greedy result
        best = _greedy_build(order_desc)

    return sorted(best)
# EVOLVE-BLOCK-END
