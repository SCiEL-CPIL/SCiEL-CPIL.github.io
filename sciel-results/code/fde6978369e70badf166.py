"""Model-and-optimize: maximize cardinality of a dissociated reciprocal subset of [1,22].

A set S is dissociated (all reciprocal subset sums distinct) iff adding each new
reciprocal never produces a subset-sum collision. We build greedily over random
orderings of [1..22] using exact integer arithmetic (scaled by LCM), with a wall-clock
deadline guard, and keep the largest valid set found.
"""
import time
import random
import math

_DEADLINE = 75.0  # seconds; safety margin before the 90 s evaluator kill
_START = time.time()

def _deadline_hit():
    return (time.time() - _START) > _DEADLINE

# Precompute LCM of 1..22
def _compute_lcm(n):
    l = 1
    for i in range(1, n + 1):
        l = l * i // math.gcd(l, i)
    return l

_LCM = _compute_lcm(22)
# Precompute scaled reciprocals: val[a] = LCM // a
_SCALED = {a: _LCM // a for a in range(1, 23)}

def _try_greedy(order):
    """Greedily add elements in `order`, keeping a set only if no sum collision.
    Returns the final list (possibly small)."""
    sums = {0}          # current subset sums (integers)
    chosen = []
    for a in order:
        r = _SCALED[a]
        # new sums are {s + r for s in sums}; must be disjoint from `sums`
        new_sums = [s + r for s in sums]
        # Check for collision using set intersection
        if set(new_sums).intersection(sums):
            continue
        # commit
        sums.update(new_sums)
        chosen.append(a)
    return chosen

def construct():
    global _START
    _START = time.time()
    random.seed()
    full = list(range(1, 23))

    # Deterministic seed baseline (known valid, 9 elements).
    seed = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best = seed
    best_k = len(seed)

    # Random restarts: greedy over random permutations.
    # Integer arithmetic allows for many more restarts.
    while not _deadline_hit():
        order = full[:]
        random.shuffle(order)
        cand = _try_greedy(order)
        if len(cand) > best_k:
            best_k = len(cand)
            best = cand
            # If we found a very large set, we can stop.
            # 11 is a strong upper bound estimate for N=22.
            if best_k >= 11:
                break

    if best is None or len(best) == 0:
        best = [1]
    return list(best)
