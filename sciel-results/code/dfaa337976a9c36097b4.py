import time
import random
from math import gcd
from functools import reduce

def _lcm_all(lst):
    return reduce(lambda a, b: a * b // gcd(a, b), lst, 1)

def _find_collision(sums, nums, indices):
    """
    Given a set of indices, check if it's dissociated.
    If not, return a pair of indices (i, j) that are part of a collision, or None if valid.
    Actually, we need to find which element to remove.
    """
    pass

def construct():
    start_time = time.time()
    deadline = start_time + 85
    
    universe = list(range(1, 23))
    N = len(universe)
    L = _lcm_all(universe)
    num = [L // e for e in universe]
    
    best_set = []
    best_len = 0
    
    def _update_best(vals):
        nonlocal best_set, best_len
        if len(vals) > best_len:
            best_len = len(vals)
            best_set = vals[:]
            
    # 1. Deterministic Greedy (Baseline)
    idx_order = sorted(range(N), key=lambda i: num[i])
    cur_sums = {0}
    cur_vals = []
    for i in idx_order:
        n = num[i]
        v = universe[i]
        has_col = False
        new_sums = set()
        for s in cur_sums:
            ns = s + n
            if ns in cur_sums:
                has_col = True
                break
            new_sums.add(ns)
        if not has_col:
            cur_sums |= new_sums
            cur_vals.append(v)
    _update_best(cur_vals)
    
    if best_len >= 17:
        return sorted(best_set)
        
    # 2. Randomized Repair Search
    # Start with random sets of size > best_len and repair them.
    
    for target_size in range(best_len + 1, N + 1):
        if (time.time() - start_time) > deadline:
            break
            
        # Try to find a valid set of this size
        # We will try a few random initializations of this size
        for attempt in range(20): # Limit attempts per size
            if (time.time() - start_time) > deadline:
                break
                
            # Pick random subset of size target_size
            current_indices = set(random.sample(range(N), target_size))
            
            # Repair loop
            # We need to check if it's dissociated.
            # If not, remove an element.
            
            # To speed up, we can just check validity repeatedly.
            # But checking validity is O(2^K).
            # For K=17, 2^17 = 131072. This is fast.
            
            while current_indices:
                # Check if current set is dissociated
                vals = [universe[i] for i in current_indices]
                nums = [num[i] for i in current_indices]
                
                sums = {0}
                valid = True
                for n in nums:
                    new_sums = set()
                    for s in sums:
                        ns = s + n
                        if ns in sums:
                            valid = False
                            break
                        new_sums.add(ns)
                    if not valid:
                        break
                    sums |= new_sums
                    
                if valid:
                    _update_best(vals)
                    break # Found a valid set of this size, try larger?
                    
                # If not valid, remove an element
                # Heuristic: remove the element with the largest numerator?
                # Or random?
                # Let's remove the one with the largest numerator (hardest to fit)
                # Or the one involved in the collision?
                # Identifying the collision element is complex.
                # Let's just remove a random element from the current set.
                # Or remove the one with max numerator to free up "space".
                
                # Let's remove the index with the largest numerator
                idx_to_remove = max(current_indices, key=lambda i: num[i])
                current_indices.remove(idx_to_remove)
                
            # End while
            
    return sorted(best_set)
