"""Reciprocal-weight 3-AP-free subset of [1,256]. Time-bounded local search + polish."""
import time
import random
from fractions import Fraction

_N = 256
_DEADLINE = 75.0  # safety margin vs 90s kill; return best found before this
_TIME0 = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _TIME0)


def _has_3ap(S):
    """True if S (set of ints) contains a nonconstant 3-term AP."""
    arr = sorted(S)
    for i in range(len(arr)):
        a = arr[i]
        for j in range(i + 1, len(arr)):
            b = arr[j]
            c = 2 * b - a
            if c > b and c in S:
                return True
    return False


def _conflicts_with(x, S):
    """Return set of y in S such that {x,y} is a 2-term segment of a 3-AP with a third in S."""
    bad = set()
    for y in S:
        # x, y, z  (x<y): need z=2y-x in S
        z = 2 * y - x
        if z > y and z in S:
            bad.add(y)
        # y, x, z  (y<x): need z=2x-y in S
        z = 2 * x - y
        if z > x and z in S:
            bad.add(y)
        # x, z, y  (x<y): x,z,y with z= (x+y)/2
        if (x + y) % 2 == 0:
            z = (x + y) // 2
            if x < z < y and z in S:
                bad.add(y)
    return bad


def _recip_sum(S):
    """Exact reciprocal sum as Fraction."""
    total = Fraction(0, 1)
    for x in S:
        total += Fraction(1, x)
    return total


def _recip_float(S):
    return sum(1.0 / x for x in S)


def _greedy(order):
    """Greedy add in given order, skipping elements that would form a 3-AP."""
    S = set()
    for x in order:
        if x in S:
            continue
        # check: does adding x create a 3-AP?
        # x with two existing: need pairs
        ok = True
        for y in S:
            z = 2 * y - x
            if z > y and z in S:
                ok = False
                break
            z = 2 * x - y
            if z > x and z in S:
                ok = False
                break
            if (x + y) % 2 == 0:
                z = (x + y) // 2
                if x < z < y and z in S:
                    ok = False
                    break
        if ok:
            S.add(x)
    return S


def _try_add(S, x):
    """Return new set if x can be added without creating a 3-AP, else None."""
    if x in S:
        return S
    for y in S:
        z = 2 * y - x
        if z > y and z in S:
            return None
        z = 2 * x - y
        if z > x and z in S:
            return None
        if (x + y) % 2 == 0:
            z = (x + y) // 2
            if x < z < y and z in S:
                return None
    return S | {x}


def _try_remove(S, x):
    if x not in S:
        return None
    return S - {x}


def _local_search(S, rng, budget_fn):
    """Iteratively improve S by single add/remove and 2-swap moves. Stops when budget_fn()<=0."""
    S = set(S)
    val = _recip_float(S)
    improved = True
    while improved and budget_fn() > 0:
        improved = False
        # single adds
        for x in range(1, _N + 1):
            if x in S:
                continue
            newS = _try_add(S, x)
            if newS is not None:
                newval = val + 1.0 / x
                if newval > val + 1e-15:
                    S = newS
                    val = newval
                    improved = True
        if budget_fn() <= 0:
            break
        # single removes (only if removing helps, i.e. to enable better adds later)
        # We do removals only as part of swaps; pure removal never helps objective.
        # 2-swap: remove a, add b (a in S, b not in S)
        removed = [x for x in S]
        candidates = [x for x in range(1, _N + 1) if x not in S]
        rng.shuffle(removed)
        rng.shuffle(candidates)
        for a in removed:
            if budget_fn() <= 0:
                break
            for b in candidates:
                # try remove a, add b
                S2 = S - {a}
                newS = _try_add(S2, b)
                if newS is not None:
                    newval = val - 1.0 / a + 1.0 / b
                    if newval > val + 1e-15:
                        S = newS
                        val = newval
                        improved = True
                        # restart loops for this a since S changed
                        break
    return S, val


def construct():
    global _TIME0
    _TIME0 = time.time()
    rng = random.Random(20240611)

    # Generate several greedy seeds
    seeds = []
    # ascending
    seeds.append(_greedy(range(1, _N + 1)))
    # descending
    seeds.append(_greedy(range(_N, 0, -1)))
    # random orders
    for _ in range(20):
        order = list(range(1, _N + 1))
        rng.shuffle(order)
        seeds.append(_greedy(order))
    # weighted random: bias toward small numbers (high reciprocal)
    for _ in range(20):
        order = list(range(1, _N + 1))
        # sort by random key biased to small
        keys = [rng.random() ** 2 + (1.0 / x) for x in order]
        order.sort(key=lambda x: -keys[order.index(x)])
        seeds.append(_greedy(order))

    # Pick best seed by reciprocal float
    best_S = None
    best_val = -1.0
    for S in seeds:
        v = _recip_float(S)
        if v > best_val:
            best_val = v
            best_S = S

    # Local search from best seed
    best_S, best_val = _local_search(best_S, rng, _time_left)

    # Polish: a few more local-search passes from the current best, with fresh shuffles
    for _ in range(5):
        if _time_left() < 3.0:
            break
        S2, v2 = _local_search(set(best_S), rng, _time_left)
        if v2 > best_val + 1e-15:
            best_S, best_val = S2, v2

    # Final safety: if deadline is near, ensure we return a valid set
    if _time_left() < 1.0:
        return sorted(best_S)

    # One more pass with a different rng seed for diversity
    rng2 = random.Random(987654321)
    S2, v2 = _local_search(set(best_S), rng2, _time_left)
    if v2 > best_val + 1e-15:
        best_S, best_val = S2, v2

    return sorted(best_S)
