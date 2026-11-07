"""Rewritten program: randomized greedy with fast incremental validity + tightening, deadline-guarded."""
import time
import random

# EVOLVE-BLOCK-START
P = {'n': 256, 'maximum_size': 30}

def _try_add(sums_set, sums_list, x):
    """Attempt to add x. sums_set: set of current nonempty subset sums.
    Returns (new_set, new_list) or None if invalid."""
    # new sums = x and (s+x for s in sums)
    new_sums = [x]
    for s in sums_list:
        ns = s + x
        if ns in sums_set:
            return None  # collision
        new_sums.append(ns)
    # check divisibility: every pair (a in sums_list, b in new_sums) and within new_sums
    # and a vs x handled by new_sums containing x
    all_pairs = []
    # new sums among themselves
    for i in range(len(new_sums)):
        for j in range(i + 1, len(new_sums)):
            a, b = new_sums[i], new_sums[j]
            if a > b:
                a, b = b, a
            if b % a == 0:
                return None
    # old vs new
    for a in sums_list:
        for b in new_sums:
            if a > b:
                a, b = b, a
            if b % a == 0:
                return None
    new_set = sums_set | set(new_sums)
    new_list = sums_list + new_sums
    return new_set, new_list

def _build_from_order(order, deadline):
    """Greedy build respecting validity incrementally."""
    sums_set = set()
    sums_list = []
    chosen = []
    for x in order:
        if time.time() > deadline:
            break
        res = _try_add(sums_set, sums_list, x)
        if res is not None:
            sums_set, sums_list = res
            chosen.append(x)
    return chosen

def _tighten(arr, deadline):
    """Try to replace elements with smaller valid ones to enable more growth."""
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(arr)):
            for x in range(1, 257):
                if x in arr:
                    continue
                if time.time() > deadline:
                    break
                trial = arr[:i] + [x] + arr[i+1:]
                # rebuild validity
                sums_set = set()
                sums_list = []
                ok = True
                for v in trial:
                    res = _try_add(sums_set, sums_list, v)
                    if res is None:
                        ok = False
                        break
                    sums_set, sums_list = res
                if ok:
                    arr = trial
                    improved = True
                    break
    return arr

def construct():
    start = time.time()
    deadline = start + 50
    best = [31, 32, 34]
    best_len = len(best)
    rng = random.Random(98765)
    pool = list(range(1, 257))

    for restart in range(500000):
        if time.time() > deadline:
            break
        k = min(len(pool), 35)
        order = rng.sample(pool, k)
        cand = _build_from_order(order, deadline)
        if len(cand) > best_len:
            best = cand
            best_len = len(cand)
            cand2 = _tighten(list(cand), deadline)
            if len(cand2) > best_len:
                best = cand2
                best_len = len(cand2)

    return [int(x) for x in best]
# EVOLVE-BLOCK-END
