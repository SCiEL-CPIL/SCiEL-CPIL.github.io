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
    
    # Strategy: Greedily build a set that maximizes sum collisions
    # Start with a seed and add elements that create the most new collisions
    
    best_set = None
    best_val = 99
    
    # Try multiple random restarts with greedy collision-maximizing construction
    for restart in range(200):
        if time.time() > deadline:
            break
        
        # Random starting point
        current = set()
        
        # Pick first element randomly
        current.add(random.randint(0, 128))
        
        # Greedily add elements that maximize collision count
        for step in range(11):
            if time.time() > deadline:
                break
            
            best_candidate = None
            best_collisions = -1
            
            # Sample candidates to speed up
            candidates = random.sample(range(0, 129), min(50, 129 - len(current)))
            for c in candidates:
                if c in current:
                    continue
                trial = list(current) + [c]
                coll = count_sum_collisions(trial)
                if coll > best_collisions:
                    best_collisions = coll
                    best_candidate = c
            
            if best_candidate is not None:
                current.add(best_candidate)
        
        if len(current) == 12:
            S = sorted(current)
            val = largest_sidon_subset_size(S)
            if val < best_val:
                best_val = val
                best_set = S[:]
        
        # Also try local search from this point
        if best_set is not None and time.time() < deadline - 10:
            # Try single-element replacements
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                i = random.randint(0, 11)
                old = best_set[i]
                new = random.randint(0, 128)
                if new == old or new in best_set:
                    continue
                trial = best_set[:]
                trial[i] = new
                val = largest_sidon_subset_size(trial)
                if val < best_val:
                    best_val = val
                    best_set = trial
    
    # If we haven't found anything, fall back to a simple set
    if best_set is None:
        best_set = list(range(12))
        best_val = largest_sidon_subset_size(best_set)
    
    return best_set
