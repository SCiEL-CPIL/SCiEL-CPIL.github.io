"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def _check_divisibility(sums_set):
    """
    Check if any two distinct values in sums_set divide one another.
    sums_set is a set of integers.
    Returns True if conflict exists, False otherwise.
    """
    sorted_sums = sorted(sums_set)
    n = len(sorted_sums)
    if n == 0:
        return False
    max_sum = sorted_sums[-1]
    
    for i in range(n):
        si = sorted_sums[i]
        if si == 0:
            continue
        multiple = 2 * si
        while multiple <= max_sum:
            if multiple in sums_set:
                return True
            multiple += si
            
    return False

def _is_valid(arr):
    """
    Check if arr is a valid set.
    1. Distinct integers in 1..256.
    2. All non-empty subset sums are unique.
    3. No two distinct subset sums divide one another.
    """
    if not arr:
        return False
    
    # Check distinctness and range
    if len(set(arr)) != len(arr):
        return False
    for x in arr:
        if not (isinstance(x, int) and 1 <= x <= 256):
            return False
            
    n = len(arr)
    sums_set = set()
    
    # Generate all non-empty subset sums
    # Using bitmask enumeration
    for mask in range(1, 1 << n):
        s = 0
        m = mask
        i = 0
        while m:
            if m & 1:
                s += arr[i]
            i += 1
            m >>= 1
            
        if s in sums_set:
            return False
        sums_set.add(s)
        
    # Check divisibility
    if _check_divisibility(sums_set):
        return False
        
    return True

def _try_greedy_extend(current, candidates, deadline):
    """
    Greedily extend current set with candidates.
    Returns the extended set.
    """
    current = list(current)
    used = set(current)
    
    for cand in candidates:
        if time.time() > deadline:
            break
        if cand in used:
            continue
            
        trial = current + [cand]
        if _is_valid(trial):
            current = trial
            used.add(cand)
            
    return current

def _incremental_check(arr):
    """
    Check validity of arr incrementally, maintaining sum set.
    Returns (is_valid, sums_set) where sums_set is None if invalid.
    """
    if not arr:
        return True, set()
    
    # Check distinctness and range
    if len(set(arr)) != len(arr):
        return False, None
    for x in arr:
        if not (isinstance(x, int) and 1 <= x <= 256):
            return False, None
            
    n = len(arr)
    sums_set = set()
    
    # Generate all non-empty subset sums
    for mask in range(1, 1 << n):
        s = 0
        m = mask
        i = 0
        while m:
            if m & 1:
                s += arr[i]
            i += 1
            m >>= 1
            
        if s in sums_set:
            return False, None
        sums_set.add(s)
        
    # Check divisibility
    if _check_divisibility(sums_set):
        return False, None
        
    return True, sums_set

def _try_greedy_extend_fast(current, candidates, deadline):
    """
    Greedily extend current set with candidates, using incremental validation.
    """
    current = list(current)
    used = set(current)
    
    # Precompute current sums
    valid, sums_set = _incremental_check(current)
    if not valid:
        return current
    
    for cand in candidates:
        if time.time() > deadline:
            break
        if cand in used:
            continue
            
        trial = current + [cand]
        valid, _ = _incremental_check(trial)
        if valid:
            current = trial
            used.add(cand)
            
    return current

def construct():
    start_time = time.time()
    # Safety margin: return well before 90s.
    deadline = start_time + 85
    
    best_set = [31, 32, 34]
    best_size = len(best_set)
    
    rng = random.Random(42)
    
    # Phase 1: Deterministic Greedy with different orderings
    orderings = [
        list(range(256, 0, -1)),  # Descending
        list(range(1, 257)),      # Ascending
        list(range(128, 257)) + list(range(1, 129)),  # Large then small
        list(range(1, 129)) + list(range(128, 257)),  # Small then large
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        current = _try_greedy_extend_fast([], order, deadline)
        if len(current) > best_size:
            best_set = current
            best_size = len(current)
    
    # Phase 2: Randomized Greedy Restarts
    large_nums = list(range(129, 257))
    small_nums = list(range(1, 129))
    
    for attempt in range(200):
        if time.time() > deadline:
            break
            
        # Create candidate list with different biases
        r = rng.random()
        if r < 0.3:
            candidates = large_nums + small_nums
        elif r < 0.6:
            candidates = small_nums + large_nums
        else:
            candidates = list(range(1, 257))
            
        rng.shuffle(candidates)
        
        current = _try_greedy_extend_fast([], candidates, deadline)
        
        if len(current) > best_size:
            best_set = current
            best_size = len(current)
    
    # Phase 3: Local Search - Try to improve best_set
    if best_size < 30 and time.time() < deadline:
        used_in_best = set(best_set)
        remaining_candidates = [x for x in range(1, 257) if x not in used_in_best]
        rng.shuffle(remaining_candidates)
        
        # Try to add any remaining candidates
        improved = _try_greedy_extend_fast(best_set, remaining_candidates, deadline)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)
        
        # Try swapping elements
        for _ in range(1000):
            if time.time() > deadline:
                break
            if not best_set:
                break
                
            idx = rng.randint(0, len(best_set) - 1)
            old_val = best_set[idx]
            new_val = rng.randint(1, 256)
            if new_val in used_in_best:
                continue
                
            trial = best_set[:]
            trial[idx] = new_val
            trial.sort()
            
            valid, _ = _incremental_check(trial)
            if valid:
                used_in_trial = set(trial)
                remaining = [x for x in range(1, 257) if x not in used_in_trial]
                rng.shuffle(remaining)
                
                extended = _try_greedy_extend_fast(trial, remaining, deadline)
                if len(extended) > best_size:
                    best_set = extended
                    best_size = len(best_set)
    
    # Final Validation
    if _is_valid(best_set):
        return list(best_set)
    else:
        return [31, 32, 34]

# EVOLVE-BLOCK-END
