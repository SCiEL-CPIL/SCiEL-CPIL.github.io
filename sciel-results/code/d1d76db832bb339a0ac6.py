"""Rewritten program: search for large reciprocal-subsum-free subsets of [1,24]."""
import time
from math import lcm
from functools import reduce

UNIVERSE = list(range(1, 25))
L = reduce(lcm, UNIVERSE)  # LCM(1..24) = 5354228880

def weights_of(S):
    """Return dict mapping element to its weight L//x."""
    return {x: L // x for x in S}

def subset_sum_exists(weights_list, target):
    """
    Check if target can be formed as a sum of a subset of weights_list.
    Uses DP with a set of achievable sums (capped for performance).
    Returns True if target is achievable.
    """
    if target <= 0:
        return target == 0
    # Quick upper bound
    if sum(weights_list) < target:
        return False
    # If any single weight equals target
    if target in weights_list:
        return True
    # DP: set of achievable sums
    achievable = {0}
    for w in weights_list:
        if w > target:
            continue
        new_sums = set()
        for s in achievable:
            ns = s + w
            if ns == target:
                return True
            if ns < target:
                new_sums.add(ns)
        achievable.update(new_sums)
        # Early termination: if we've found target
        if target in achievable:
            return True
    return target in achievable

def is_valid(S):
    """
    Check if set S is reciprocal-subsum-free.
    For every x in S, L/x must NOT be representable as a subset sum
    of {L/y for y in S, y != x}.
    """
    if len(S) <= 1:
        return True
    S_list = list(S)
    w = {x: L // x for x in S_list}
    for x in S_list:
        target = w[x]
        others = [w[y] for y in S_list if y != x]
        if subset_sum_exists(others, target):
            return False
    return True

def greedy_construct():
    """
    Greedy: try adding elements in order of decreasing weight (i.e., increasing x).
    Larger x have smaller weights, making them harder to form as subset sums.
    """
    S = []
    # Try adding from largest to smallest (smallest weights first)
    for x in sorted(UNIVERSE, reverse=True):
        candidate = S + [x]
        if is_valid(candidate):
            S.append(x)
    return S

def greedy_construct_forward():
    """Greedy from smallest to largest."""
    S = []
    for x in sorted(UNIVERSE):
        candidate = S + [x]
        if is_valid(candidate):
            S.append(x)
    return S

def local_search(S, deadline):
    """
    Hill climbing: try to add elements not in S, or swap elements to increase size.
    """
    S = list(S)
    best = S
    best_size = len(S)
    while time.time() < deadline:
        improved = False
        # Try adding each missing element
        for x in UNIVERSE:
            if x not in S:
                candidate = S + [x]
                if is_valid(candidate):
                    S = candidate
                    best = S[:]
                    best_size = len(S)
                    improved = True
                    break
        if improved:
            continue
        # Try swaps: remove one, add one different
        for i, xi in enumerate(S):
            for xj in UNIVERSE:
                if xj in S:
                    continue
                candidate = [y for j, y in enumerate(S) if j != i] + [xj]
                if len(candidate) == len(S) and is_valid(candidate):
                    # Only accept if it helps future additions (heuristic: keep)
                    # For now, just check if we can then add more
                    S = candidate
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return best

def random_restart_construct(deadline):
    """
    Multiple random-order greedy constructions, keep best.
    """
    import random
    best = None
    best_size = 0
    while time.time() < deadline:
        random.shuffle(UNIVERSE)
        S = []
        for x in UNIVERSE:
            candidate = S + [x]
            if is_valid(candidate):
                S.append(x)
        if len(S) > best_size:
            best_size = len(S)
            best = S[:]
    return best if best else []

def construct():
    start = time.time()
    deadline = start + 80  # Safety margin before 90s kill
    
    # Phase 1: Deterministic greedy (fast)
    S1 = greedy_construct()
    S2 = greedy_construct_forward()
    best = S1 if len(S1) >= len(S2) else S2
    
    # Phase 2: Local search on best
    if time.time() < deadline:
        best = local_search(best, deadline)
    
    # Phase 3: Random restarts if time remains
    if time.time() < deadline - 5:
        rand_best = random_restart_construct(deadline)
        if len(rand_best) > len(best):
            best = rand_best
    
    # Final: try one more local search pass
    if time.time() < deadline - 2:
        best = local_search(best, deadline)
    
    # Ensure nonempty
    if not best:
        best = [24]
    
    return sorted(best)
