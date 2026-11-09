import time
import random
from itertools import combinations

def is_sidon(subset):
    sums = set()
    n = len(subset)
    for i in range(n):
        for j in range(i, n):
            s = subset[i] + subset[j]
            if s in sums:
                return False
            sums.add(s)
    return True

def largest_sidon_subset_size(S):
    S = sorted(S)
    n = len(S)
    # Try from largest k down
    for k in range(n, 0, -1):
        for combo in combinations(S, k):
            if is_sidon(combo):
                return k
    return 1

def count_sum_collisions(S):
    """Count how many pair sums are duplicated (higher = more collisions = harder for Sidon)"""
    sums = {}
    n = len(S)
    for i in range(n):
        for j in range(i, n):
            s = S[i] + S[j]
            if s in sums:
                sums[s] += 1
            else:
                sums[s] = 1
    # Total collisions = sum of (count-1) for each sum that appears more than once
    collisions = sum(v - 1 for v in sums.values() if v > 1)
    return collisions

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    best_set = None
    best_collisions = -1
    
    # Strategy: Generate dense sets (APs) and improve via collision maximization
    # APs are expected to have the smallest largest Sidon subsets.
    
    # Initial candidates: APs with different steps
    candidates = []
    for d in range(1, 10):
        for start in range(0, 128 - 11*d + 1, d): # Start such that we fit 12 elements
            # Actually, just start=0 is enough for isomorphism, but let's vary to be safe
            pass
            
    # Just use a few representative dense structures
    # 1. Standard AP
    candidates.append(list(range(12)))
    # 2. AP with step 2
    candidates.append([2*i for i in range(12)])
    # 3. Random dense cluster (start + random small offsets)
    for _ in range(20):
        start = random.randint(0, 110)
        offsets = random.sample(range(15), 12) # Small range to keep dense
        # Ensure distinct
        s = set(start + o for o in offsets)
        if len(s) == 12:
            candidates.append(sorted(s))
            
    # Evaluate candidates using the fast collision proxy
    for c in candidates:
        if len(c) != 12:
            continue
        # Check bounds
        if any(x < 0 or x > 128 for x in c):
            continue
        coll = count_sum_collisions(c)
        if coll > best_collisions:
            best_collisions = coll
            best_set = c[:]
            
    # Local Search using Collision Proxy
    # This is much faster than exact Sidon solving
    if best_set is not None:
        current_set = best_set[:]
        current_coll = best_collisions
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(12):
                if time.time() > deadline - 5:
                    break
                old_val = current_set[i]
                # Try swapping with nearby values
                for offset in range(1, 20):
                    for new_val in [old_val + offset, old_val - offset]:
                        if new_val < 0 or new_val > 128:
                            continue
                        if new_val in current_set:
                            continue
                        trial = current_set[:]
                        trial[i] = new_val
                        trial_coll = count_sum_collisions(trial)
                        if trial_coll > current_coll:
                            current_coll = trial_coll
                            current_set = trial
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
                    
        best_set = current_set
        best_collisions = current_coll

    # Fallback
    if best_set is None:
        best_set = list(range(12))
        
    return best_set
