"""Improved program: cardinality-indexed subset-sum DP to maximize set size."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

DEADLINE_SECONDS = 55.0  # safety margin before 60s kill


def _subset_sums_by_cardinality(S):
    """Return list: sums_by_k[k] = set of sums achievable with exactly k elements."""
    n = len(S)
    sums_by_k = [set() for _ in range(n + 1)]
    sums_by_k[0].add(0)
    for x in S:
        # Iterate k in reverse to avoid using x more than once
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


def _is_valid_incremental(S, candidate):
    """
    Check if adding candidate to S keeps the set valid.
    Uses incremental DP: compute sums_by_k for S, then check if adding candidate
    creates any cross-cardinality collision.
    """
    n = len(S)
    # Compute sums_by_k for current S
    sums_by_k = [set() for _ in range(n + 1)]
    sums_by_k[0].add(0)
    for x in S:
        for k in range(n, 0, -1):
            if sums_by_k[k - 1]:
                for s in sums_by_k[k - 1]:
                    sums_by_k[k].add(s + x)
    
    # Now add candidate: new sums_by_k'[k] = sums_by_k[k] ∪ {s + candidate for s in sums_by_k[k-1]}
    # We need to check: for each k, the new sums at cardinality k must not overlap with sums at any other cardinality
    # More precisely: for each k, new_sums_at_k = sums_by_k[k] ∪ {s + candidate for s in sums_by_k[k-1]}
    # We need: for all k != j, new_sums_at_k ∩ new_sums_at_j = ∅
    
    # Build the full new_sums_by_k
    new_sums_by_k = [set() for _ in range(n + 2)]
    new_sums_by_k[0].add(0)
    for k in range(1, n + 1):
        # Inherited from S
        new_sums_by_k[k] = set(sums_by_k[k])
        # New sums from adding candidate to subsets of size k-1
        if sums_by_k[k - 1]:
            for s in sums_by_k[k - 1]:
                new_sums_by_k[k].add(s + candidate)
    # new_sums_by_k[n+1] = {s + candidate for s in sums_by_k[n]}
    if sums_by_k[n]:
        for s in sums_by_k[n]:
            new_sums_by_k[n + 1].add(s + candidate)
    
    # Check for cross-cardinality collisions
    sum_to_k = defaultdict(set)
    for k in range(n + 2):
        for s in new_sums_by_k[k]:
            sum_to_k[s].add(k)
    for s, ks in sum_to_k.items():
        if len(ks) > 1:
            return False
    return True


def _greedy_build(candidates, deadline):
    """Greedily build a valid set from candidates list."""
    current = []
    for x in candidates:
        if time.time() > deadline:
            break
        if len(current) >= 60:
            break
        # Check if adding x is valid
        if _is_valid_incremental(current, x):
            current.append(x)
    return current


def construct():
    t_start = time.time()
    deadline = t_start + DEADLINE_SECONDS
    
    best_set = []
    best_size = 0
    
    # Strategy 1: Deterministic ascending
    candidates = list(range(1, 257))
    current = _greedy_build(candidates, deadline)
    if len(current) > best_size:
        best_size = len(current)
        best_set = current[:]
    
    # Strategy 2: Deterministic descending
    candidates = list(range(256, 0, -1))
    current = _greedy_build(candidates, deadline)
    if len(current) > best_size:
        best_size = len(current)
        best_set = current[:]
    
    # Strategy 3: Random restarts with various biases
    random.seed(42)
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        # Various randomization strategies
        if attempt % 3 == 0:
            candidates = list(range(1, 257))
            random.shuffle(candidates)
        elif attempt % 3 == 1:
            # Bias toward smaller values
            candidates = list(range(1, 129))
            random.shuffle(candidates)
            candidates += list(range(129, 257))
            random.shuffle(candidates)
        else:
            # Mix of small and large
            small = list(range(1, 65))
            large = list(range(192, 257))
            random.shuffle(small)
            random.shuffle(large)
            candidates = small + large
            random.shuffle(candidates)
        
        current = _greedy_build(candidates, deadline)
        if len(current) > best_size:
            best_size = len(current)
            best_set = current[:]
        
        # If we've found a large set, try to improve it further
        if best_size >= 20:
            # Try adding remaining candidates not in best_set
            remaining = [x for x in range(1, 257) if x not in best_set]
            random.shuffle(remaining)
            improved = best_set[:]
            for x in remaining:
                if time.time() > deadline:
                    break
                if len(improved) >= 60:
                    break
                if _is_valid_incremental(improved, x):
                    improved.append(x)
            if len(improved) > best_size:
                best_size = len(improved)
                best_set = improved[:]
    
    return list(best_set)
# EVOLVE-BLOCK-END
