import time
import random
from itertools import combinations

# LCM of 1..22
_LCM = 232792560

def _is_valid(subset):
    """
    Check if all subset sums of reciprocals are distinct.
    Equivalent to checking if all subset sums of LCM/i are distinct.
    """
    if not subset:
        return True
    
    # Convert to integer weights
    weights = [_LCM // i for i in subset]
    k = len(weights)
    
    # Generate all subset sums incrementally
    # Use a set to store sums, check for collisions early
    sums = {0}
    for w in weights:
        # Add w to all existing sums
        new_sums = set()
        for s in sums:
            ns = s + w
            if ns in sums:
                return False
            new_sums.add(ns)
        sums |= new_sums
        
    return True

def _is_valid_fast(subset, existing_sums=None):
    """
    Check if adding subset to existing_sums keeps all sums distinct.
    If existing_sums is None, start from {0}.
    """
    if not subset:
        return True, existing_sums if existing_sums is not None else {0}
    
    if existing_sums is None:
        sums = {0}
    else:
        sums = set(existing_sums)
    
    weights = [_LCM // i for i in subset]
    
    for w in weights:
        new_sums = set()
        for s in sums:
            ns = s + w
            if ns in sums:
                return False, sums
            new_sums.add(ns)
        sums |= new_sums
        
    return True, sums

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Start with a known good seed
    best_subset = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_subset)
    best_sums = None
    
    # Precompute weights for quick access
    lcm_weights = {i: _LCM // i for i in range(1, 23)}
    
    # Strategy 1: Greedy with multiple random restarts
    # Try different orderings to find larger valid sets
    for _ in range(5000):
        if time.time() > deadline:
            break
            
        # Random permutation of 1..22
        candidates = list(range(1, 23))
        random.shuffle(candidates)
        
        current = []
        current_sums = {0}
        for c in candidates:
            # Check if adding c is valid using the fast method
            valid, new_sums = _is_valid_fast([c], current_sums)
            if valid:
                current.append(c)
                current_sums = new_sums
                
        if len(current) > best_size:
            best_size = len(current)
            best_subset = current
            best_sums = current_sums
            
            # If we found a very large set, try to extend it
            if best_size >= 15:
                # Try to add more elements to this set
                remaining = [x for x in range(1, 23) if x not in current]
                for c in remaining:
                    valid, new_sums = _is_valid_fast([c], best_sums)
                    if valid:
                        current.append(c)
                        best_sums = new_sums
                        best_size = len(current)
                        best_subset = current
                        if best_size >= 18:
                            break

    # Strategy 2: Local search - try to swap elements to fix invalid sets
    # Start from random larger sets and try to repair them
    for _ in range(3000):
        if time.time() > deadline:
            break
            
        # Start with a random set of size 14-16
        target_size = random.choice([14, 15, 16])
        perm = list(range(1, 23))
        random.shuffle(perm)
        current = perm[:target_size]
        
        # Try to make this set valid by swapping elements
        improved = True
        max_swaps = 20
        swap_count = 0
        while improved and swap_count < max_swaps:
            improved = False
            swap_count += 1
            
            # Check if current is valid
            valid, current_sums = _is_valid_fast(current)
            if valid:
                # Try to add more
                remaining = [x for x in range(1, 23) if x not in current]
                for c in remaining:
                    valid2, new_sums = _is_valid_fast([c], current_sums)
                    if valid2:
                        current.append(c)
                        current_sums = new_sums
                        improved = True
                        break
                if len(current) > best_size:
                    best_size = len(current)
                    best_subset = current
                    best_sums = current_sums
                break
            else:
                # Not valid, try to fix by swapping
                # Find which element causes the most conflicts
                # Simple approach: try swapping each element with a random other
                for i in range(len(current)):
                    if time.time() > deadline:
                        break
                    # Try swapping current[i] with a random element not in current
                    others = [x for x in range(1, 23) if x not in current]
                    if not others:
                        continue
                    swap_with = random.choice(others)
                    new_current = current[:i] + [swap_with] + current[i+1:]
                    valid2, _ = _is_valid_fast(new_current)
                    if valid2:
                        current = new_current
                        improved = True
                        break

    # Strategy 3: Try to extend the best found set
    if best_sums is not None and len(best_subset) < 22:
        remaining = [x for x in range(1, 23) if x not in best_subset]
        for c in remaining:
            if time.time() > deadline:
                break
            valid, new_sums = _is_valid_fast([c], best_sums)
            if valid:
                best_subset.append(c)
                best_sums = new_sums
                best_size = len(best_subset)

    return best_subset
