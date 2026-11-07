"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools

_DEADLINE = 80.0  # safety margin before 90s kill

# Primes up to 40
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
_NP = len(_PRIMES)

def _prime_vec(n):
    v = [0] * _NP
    for i, p in enumerate(_PRIMES):
        while n % p == 0:
            v[i] += 1
            n //= p
    return tuple(v)

_CANDS = list(range(2, 41))
_CAND_VEC = {a: _prime_vec(a) for a in _CANDS}

def _subset_sums_distinct(vecs):
    """Check that all 2^k subset sums of exponent vectors are distinct."""
    k = len(vecs)
    if k == 0:
        return True
    # Build all subset sums incrementally
    sums = {()}
    for v in vecs:
        new = set()
        for s in sums:
            ns = tuple(s[i] + v[i] for i in range(_NP))
            new.add(ns)
        if len(sums) + len(new) != len(sums | new):
            return False
        sums = sums | new
    return len(sums) == (1 << k)

def _valid(vecs):
    return _subset_sums_distinct(vecs)

def _greedy(order, current_vecs):
    """Try to add elements in given order."""
    chosen = list(current_vecs)
    chosen_nums = []
    # track which numbers correspond
    for a in order:
        v = _CAND_VEC[a]
        if _valid(chosen + [v]):
            chosen.append(v)
            chosen_nums.append(a)
    return chosen_nums

def _local_search(chosen_nums, deadline):
    """Improve by trying swaps: remove one, add another."""
    chosen = set(chosen_nums)
    vecs = [_CAND_VEC[a] for a in chosen]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for a in list(chosen):
            for b in _CANDS:
                if b in chosen:
                    continue
                new_set = (chosen - {a}) | {b}
                if len(new_set) != len(chosen):
                    continue
                nv = [_CAND_VEC[x] for x in new_set]
                if _valid(nv):
                    # accept if same size (no gain) — only useful if we can then add more; 
                    # skip pure swaps, focus on additions
                    pass
    # Try to add any remaining element
    for b in _CANDS:
        if b not in chosen and time.time() >= deadline:
            break
        if _valid(vecs + [_CAND_VEC[b]]):
            chosen.add(b)
            vecs.append(_CAND_VEC[b])
            improved = True
    return sorted(chosen)

def _full_greedy_with_additions(order):
    """Greedy add in order, then try to fill with any remaining."""
    chosen = []
    vecs = []
    for a in order:
        if _valid(vecs + [_CAND_VEC[a]]):
            chosen.append(a)
            vecs.append(_CAND_VEC[a])
    # fill
    for b in _CANDS:
        if b not in chosen:
            if _valid(vecs + [_CAND_VEC[b]]):
                chosen.append(b)
                vecs.append(_CAND_VEC[b])
    return chosen

def construct():
    start = time.time()
    deadline = start + _DEADLINE
    best = []

    # Multiple orderings
    orderings = []
    # ascending
    orderings.append(list(range(2, 41)))
    # descending
    orderings.append(list(range(40, 1, -1)))
    # primes first
    primes_in_range = [a for a in range(2, 41) if all(a % d for d in range(2, int(a**0.5)+1))]
    non_primes = [a for a in range(2, 41) if a not in primes_in_range]
    orderings.append(primes_in_range + non_primes)
    # small primes first
    orderings.append(sorted(range(2, 41), key=lambda x: _CAND_VEC[x][0] + _CAND_VEC[x][1]))
    # reverse of that
    orderings.append(sorted(range(2, 41), key=lambda x: -(_CAND_VEC[x][0] + _CAND_VEC[x][1])))

    for order in orderings:
        if time.time() > deadline:
            break
        res = _full_greedy_with_additions(order)
        if len(res) > len(best):
            best = res
        # local search / fill
        res2 = _local_search(res, deadline)
        if len(res2) > len(best):
            best = res2

    # Try random orderings
    import random
    rng = random.Random(42)
    for _ in range(200):
        if time.time() > deadline:
            break
        order = list(range(2, 41))
        rng.shuffle(order)
        res = _full_greedy_with_additions(order)
        if len(res) > len(best):
            best = res
        res2 = _local_search(res, deadline)
        if len(res2) > len(best):
            best = res2

    return [int(x) for x in best]
# EVOLVE-BLOCK-END
