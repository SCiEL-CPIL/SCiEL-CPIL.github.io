"""Improved program: cardinality-indexed subset-sum DP to maximize set size."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

DEADLINE_SECONDS = 55.0  # safety margin before 60s kill

def _subset_sums_by_cardinality(S):
    """Return dict: cardinality -> set of achievable sums."""
    n = len(S)
    # sums_by_k[k] = set of sums achievable with exactly k elements
    sums_by_k = [set() for _ in range(n + 1)]
    sums_by_k[0].add(0)
    for x in S:
        for k in range(n, 0, -1):
            if sums_by_k[k - 1]:
                for s in sums_by_k[k - 1]:
                    sums_by_k[k].add(s + x)
    return sums_by_k

def _is_valid(S):
    """Check that no two subsets of different cardinalities have the same sum."""
    sums_by_k = _subset_sums_by_cardinality(S)
    # Build a map: sum -> set of cardinalities that achieve it
    sum_to_k = defaultdict(set)
    for k, sums in enumerate(sums_by_k):
        for s in sums:
            sum_to_k[s].add(k)
    for s, ks in sum_to_k.items():
        if len(ks) > 1:
            return False
    return True

def _try_add(S, candidate, time_limit):
    """Try adding candidate to S; return True if valid."""
    if time.time() > time_limit:
        return False
    new_S = S + [candidate]
    return _is_valid(new_S)

def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS
    
    # Strategy: greedily build a set using DP-based validation
    # Start with small values to maximize room for many elements
    best_set = []
    
    # Try multiple random orderings to find a larger valid set
    best_size = 0
    
    # First, try a deterministic approach: sort candidates by value
    # and greedily add if valid
    for attempt in range(200):
        if time.time() > deadline:
            break
        
        if attempt == 0:
            # Deterministic: try smallest values first
            candidates = list(range(1, 257))
        elif attempt == 1:
            # Try larger values first (sometimes helps)
            candidates = list(range(256, 0, -1))
        else:
            candidates = list(range(1, 257))
            random.shuffle(candidates)
        
        current = []
        for x in candidates:
            if time.time() > deadline:
                break
            if len(current) >= 60:
                break
            # Quick check: can we add x?
            # Use incremental DP check
            new_S = current + [x]
            if _is_valid(new_S):
                current = new_S
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
    
    # Return as list of Python ints
    return list(best_set)
# EVOLVE-BLOCK-END
