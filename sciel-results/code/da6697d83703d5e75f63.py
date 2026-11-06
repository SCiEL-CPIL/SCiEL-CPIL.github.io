"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _is_sidon(subset):
    """
    Check if a subset has all unordered pair sums (including diagonals) unique.
    Input: list of ints
    Output: bool
    """
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def _max_sidon_size(S):
    """
    Compute the size of the largest Sidon subset of S.
    S is a list of 12 distinct integers.
    """
    n = len(S)
    # We check sizes from n down to 1.
    # If we find a Sidon subset of size k, that's the max for this S.
    for k in range(n, 0, -1):
        # Iterate through combinations of size k
        # Optimization: If k is large, few combinations. If k is small, many.
        # But we stop as soon as we find one.
        # For k=n, 1 check.
        # For k=n-1, n checks.
        # etc.
        for comb in combinations(S, k):
            if _is_sidon(comb):
                return k
    return 0

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 90s, we aim for 85s max.
    deadline = start_time + 85.0
    
    # Initial candidates
    # 1. Standard AP
    candidates = [
        list(range(12)),
        list(range(0, 128, 10))[:12],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        [0, 1, 3, 7, 15, 31, 63, 127, 2, 5, 11, 23] # Mixed
    ]
    
    # Add some random initializations
    random.seed(42)
    for _ in range(20):
        candidates.append(random.sample(range(129), 12))
        
    best_set = None
    best_score = float('inf')
    
    # Evaluate initial candidates
    for cand in candidates:
        if time.time() > deadline:
            break
        score = _max_sidon_size(cand)
        if score < best_score:
            best_score = score
            best_set = cand[:]
            
    if best_set is None:
        return list(range(12))
        
    current_set = best_set[:]
    current_score = best_score
    
    # Local Search
    # We want to minimize the max Sidon size.
    # Move: Replace one element with a random one.
    
    while time.time() < deadline:
        # Pick an index to modify
        idx = random.randint(0, 11)
        old_val = current_set[idx]
        
        # Pick a new value
        # Bias towards values that might create collisions?
        # Hard to bias directly. Random is fine.
        new_val = random.randint(0, 128)
        
        # Ensure distinctness
        if new_val == old_val:
            continue
        if new_val in current_set:
            continue
            
        new_set = current_set[:]
        new_set[idx] = new_val
        new_set.sort() # Sorting helps combinations be consistent, though not strictly necessary for set logic
        
        new_score = _max_sidon_size(new_set)
        
        if new_score < best_score:
            best_score = new_score
            best_set = new_set[:]
            current_set = new_set[:]
            current_score = new_score
            if best_score <= 3:
                # 3 is very good. 2 is impossible for 12 distinct ints? 
                # Actually, if every triple is an AP, max size is 2.
                # Can we have 12 distinct ints where every triple is an AP? No.
                # But if we get to 3, we are likely near optimal.
                # We can continue to see if 2 is possible, but it's unlikely.
                pass
        elif new_score == best_score:
            # Accept with some probability to explore the plateau
            if random.random() < 0.3:
                current_set = new_set[:]
                current_score = new_score
        else:
            # Accept worse moves rarely to escape local minima
            if random.random() < 0.05:
                current_set = new_set[:]
                current_score = new_score

    return best_set

# EVOLVE-BLOCK-END
