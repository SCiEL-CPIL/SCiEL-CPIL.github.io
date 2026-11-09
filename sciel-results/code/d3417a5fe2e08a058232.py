import time
import random
from math import lcm
from functools import reduce

UNIVERSE = list(range(1, 25))
L = reduce(lcm, UNIVERSE)
WEIGHTS = {i: L // i for i in UNIVERSE}

def get_subset_sums(weights):
    """
    Returns a set of all possible subset sums of the given list of weights.
    """
    sums = {0}
    for w in weights:
        new_sums = set()
        for s in sums:
            ns = s + w
            new_sums.add(ns)
        sums.update(new_sums)
    return sums

def is_valid_subset(S):
    """
    Check if the set S is reciprocal-subsum-free.
    Optimization: Compute all subset sums of S.
    If any weight w_x is in the subset sums, then w_x = sum(w_y for y in T).
    Since all weights are positive, T cannot contain x.
    Thus, the condition "subset of OTHERS" is automatically satisfied if w_x is a subset sum of S.
    """
    if not S:
        return True
    
    S_list = list(S)
    weights = [WEIGHTS[x] for x in S_list]
    all_sums = get_subset_sums(weights)
    
    for x in S_list:
        if WEIGHTS[x] in all_sums:
            return False
            
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85.0 # 5s margin
    
    best_S = set()
    
    # Randomized Greedy Construction
    # We try many random orderings of the universe.
    # For each ordering, we greedily add elements if they don't break validity.
    
    # To speed up, we can pre-shuffle and use a fast validity check.
    # The validity check with precomputed sums is O(2^N) in worst case but N is small.
    # Actually, the number of subset sums can be large, but for N=24, it's manageable.
    # However, doing this for every addition step in greedy might be slow.
    # Let's estimate: N=24. Greedy adds up to 24 elements.
    # Each step checks validity of S (size k).
    # get_subset_sums for k elements has at most 2^k sums.
    # Sum of 2^k for k=1..24 is approx 2^25 = 33 million.
    # This is too slow for many restarts in 85s.
    
    # Alternative: Incremental subset sums?
    # Or, just use the original check for the greedy step, which is faster on average?
    # The original check: For each x in S, check subset sum of others.
    # If S is small, it's fast.
    
    # Let's stick to the original validity check for the greedy steps, but optimize the search.
    # Or, use the "all_sums" check only at the end of a greedy pass?
    # No, we need to check validity to decide if we add.
    
    # Let's try to optimize the original check.
    # The bottleneck is `find_conflicts` or `is_valid_subset`.
    
    # Revised Plan:
    # 1. Use randomized greedy add with the *original* `is_valid_subset` (which checks per-element).
    #    This is likely fast enough for small S.
    # 2. Use multiple restarts.
    
    def is_valid_fast(S):
        S_list = list(S)
        n = len(S_list)
        if n == 0:
            return True
        for i, x in enumerate(S_list):
            target = WEIGHTS[x]
            others = [WEIGHTS[y] for y in S_list if y != x]
            # Quick sum check
            if sum(others) < target:
                continue
            # Subset sum DP with pruning
            possible_sums = {0}
            for w in others:
                if w > target:
                    continue
                new_sums = set()
                for s in possible_sums:
                    ns = s + w
                    if ns <= target:
                        new_sums.add(ns)
                possible_sums.update(new_sums)
                if target in possible_sums:
                    return False
        return True

    # Run many restarts
    for _ in range(200):
        if time.time() > deadline:
            break
            
        random.shuffle(UNIVERSE)
        S = set()
        
        for x in UNIVERSE:
            new_S = S | {x}
            if is_valid_fast(new_S):
                S = new_S
        
        if len(S) > len(best_S):
            best_S = set(S)
            
    # Local Search / Swaps on the best found set
    # Try to improve best_S by swapping elements
    # If we can't increase size, we just return best_S.
    # Can we increase size?
    # If best_S is maximal, we can't add any element.
    # We might need to remove one to add two?
    # This is complex. Let's just rely on greedy restarts.
    
    # One more pass: Try to add any remaining element to best_S
    remaining = [e for e in range(1, 25) if e not in best_S]
    for e in remaining:
        new_S = best_S | {e}
        if is_valid_fast(new_S):
            best_S = new_S
            # If we added one, check others again?
            # Just do a simple loop.
            
    return list(best_S)
