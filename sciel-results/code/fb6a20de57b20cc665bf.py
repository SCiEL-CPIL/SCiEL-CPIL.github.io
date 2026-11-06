"""Fast search for large dissociated reciprocal subsets of [1,22] using integer arithmetic."""
import time
import random

# EVOLVE-BLOCK-START

_LCM = 232792560
_INV = {x: _LCM // x for x in range(1, 23)}
_ALL = list(range(1, 23))

def _greedy_order(order, deadline):
    """
    Given an ordering of candidates, greedily add elements maintaining dissociation.
    Returns (set_of_elements, set_of_subset_sums).
    Uses fast incremental collision check.
    """
    sums = {0}
    chosen = set()
    for x in order:
        if x in chosen:
            continue
        inv = _INV[x]
        # collision check: does shifting sums by inv hit sums?
        coll = False
        for s in sums:
            if (s + inv) in sums:
                coll = True
                break
        if coll:
            continue
        # accept
        # update sums: add s+inv for all s
        for s in list(sums):
            sums.add(s + inv)
        chosen.add(x)
        if time.time() > deadline:
            break
    return chosen, sums

def _complete(chosen, sums, deadline):
    """Try to add more elements to chosen in random order until no more fit."""
    remaining = [x for x in _ALL if x not in chosen]
    improved = True
    while improved and time.time() < deadline:
        improved = False
        random.shuffle(remaining)
        for x in remaining:
            if x in chosen:
                continue
            inv = _INV[x]
            coll = False
            for s in sums:
                if (s + inv) in sums:
                    coll = True
                    break
            if not coll:
                for s in list(sums):
                    sums.add(s + inv)
                chosen.add(x)
                remaining.remove(x)
                improved = True
                break
    return chosen, sums

def construct():
    start = time.time()
    deadline = start + 50  # safety margin under 60s (evaluator kills at 90s)

    best = set()
    best_size = 0

    # Warm start: a few deterministic-ish orders
    for seed_order in range(30):
        if time.time() > deadline:
            break
        order = _ALL[:]
        random.shuffle(order)
        chosen, sums = _greedy_order(order, deadline)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best = set(chosen)

    # Main loop: random greedy + completion
    while time.time() < deadline - 2:
        order = _ALL[:]
        random.shuffle(order)
        chosen, sums = _greedy_order(order, deadline - 2)
        # completion
        chosen, sums = _complete(chosen, sums, deadline - 1)
        if len(chosen) > best_size:
            best_size = len(chosen)
            best = set(chosen)

    if not best:
        best = {1}

    return sorted(best)

# EVOLVE-BLOCK-END
