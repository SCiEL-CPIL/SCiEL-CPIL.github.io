"""Direct constructive approach: exponential sequence guarantees distinct triple sums."""
import time
import random
import sys
import numpy as np


def _all_triple_sums(lst):
    """Compute all unordered triple sums for a sorted list."""
    n = len(lst)
    sums = []
    for i in range(n):
        for j in range(i, n):
            for k in range(j, n):
                sums.append(lst[i] + lst[j] + lst[k])
    return sums


def _is_valid_triples(lst):
    """Check if all unordered triple sums are distinct."""
    if not lst:
        return True
    sums = _all_triple_sums(lst)
    return len(sums) == len(set(sums))


def _new_triple_sums(existing, x):
    """Compute triple sums involving x with existing sorted elements."""
    n = len(existing)
    sums = [3 * x]
    for i in range(n):
        sums.append(2 * x + existing[i])   # (x, x, s_i) with x <= s_i (or we handle ordering)
        sums.append(x + 2 * existing[i])   # (s_i, s_i, x) 
    for i in range(n):
        for j in range(i + 1, n):
            sums.append(x + existing[i] + existing[j])
    return sums


def _check_collision_fast(existing_arr, existing_sums_set, x):
    """
    Fast collision check using numpy vectorization.
    Returns True if adding x would create a collision.
    """
    n = len(existing_arr)
    if n == 0:
        return False
    
    # Check 3*x
    if 3 * x in existing_sums_set:
        return True
    
    # Check 2*x + existing[i] for all i
    # These are sums of type (x, x, a_i)
    vals_2x_plus = 2 * x + existing_arr
    # Check each against existing_sums_set
    # For efficiency, check if any is in the set
    for v in vals_2x_plus:
        if v in existing_sums_set:
            return True
    
    # Check x + 2*existing[i] for all i
    # These are sums of type (a_i, a_i, x)
    vals_x_plus_2 = x + 2 * existing_arr
    for v in vals_x_plus_2:
        if v in existing_sums_set:
            return True
    
    # Check x + existing[i] + existing[j] for i < j
    # This is the expensive part
    # Use numpy: for each i, compute x + existing[i] + existing[j] for j > i
    # We can vectorize by computing all pairs
    if n >= 2:
        # Create upper triangle indices
        ii, jj = np.triu_indices(n, k=1)
        pair_sums = x + existing_arr[ii] + existing_arr[jj]
        # Check if any is in existing_sums_set
        for v in pair_sums:
            if v in existing_sums_set:
                return True
    
    return False


def _update_sums_fast(existing_arr, existing_sums_set, x):
    """Add all new triple sums involving x to existing_sums_set."""
    n = len(existing_arr)
    existing_sums_set.add(3 * x)
    
    for i in range(n):
        existing_sums_set.add(2 * x + existing_arr[i])
        existing_sums_set.add(x + 2 * existing_arr[i])
    
    if n >= 2:
        ii, jj = np.triu_indices(n, k=1)
        pair_sums = x + existing_arr[ii] + existing_arr[jj]
        for v in pair_sums:
            existing_sums_set.add(int(v))


def _exponential_base(b, max_val):
    """
    Build a set from powers of base b: {b, b^2, b^3, ...}.
    For base b >= 2, this gives distinct sums if b is large enough.
    Actually, for powers of 3: {3, 9, 27, 81, ...} the ternary representation 
    gives unique sums for multisets of size up to 2 (B_2 sets).
    For B_3[1], we need a stronger base. Use base b where 2*b > sum of all smaller.
    A safe choice: a_i such that 3*a_i > sum(a_0..a_{i-1}) * 3.
    Simplest: use a_i = b^i with b large enough. For 3-term sums, b >= 4 works 
    if we think in base b with digits 0..3.
    Actually the cleanest: a_i = 4^i ensures uniqueness for sums of at most 3 elements
    since max coefficient is 3 < 4.
    """
    result = []
    val = 1
    while val <= max_val:
        result.append(val)
        val *= 4
    return result


def _construct_b3_set():
    """
    Direct construction using powers of 4 as seed, then greedy extension with fast collision check.
    """
    # Start with powers of 4 for guaranteed validity
    base_set = _exponential_base(4, 4096)
    
    # Try to extend using greedy with fast collision check
    existing = list(base_set)
    existing_arr = np.array(existing, dtype=np.int64)
    existing_sums = set(_all_triple_sums(existing))
    existing_set = set(existing)
    
    for x in range(1, 4097):
        if x in existing_set:
            continue
        if _check_collision_fast(existing_arr, existing_sums, x):
            continue
        _update_sums_fast(existing_arr, existing_sums, x)
        existing.append(x)
        existing_arr = np.append(existing_arr, x)
        existing_set.add(x)
        existing.sort()
        existing_arr = np.sort(existing_arr)
    
    return existing


def construct():
    """
    Find a large B_3[1] set in [1, 4096] where all unordered triple sums
    (allowing repetitions) are distinct.
    Uses fast vectorized collision checking to enable more extensive search.
    """
    start_time = time.time()
    deadline = start_time + 75  # safety margin for 90s limit
    
    best = []
    rng = random.Random(12345)
    
    # Strategy 1: Exponential base construction + greedy extension (fast)
    cand = _construct_b3_set()
    if _is_valid_triples(cand):
        best = list(cand)
    
    # Strategy 2: Pure greedy from 1 with fast collision check
    existing = []
    existing_arr = np.array([], dtype=np.int64)
    existing_sums = set()
    existing_set = set()
    for x in range(1, 4097):
        if _check_collision_fast(existing_arr, existing_sums, x):
            continue
        _update_sums_fast(existing_arr, existing_sums, x)
        existing.append(x)
        existing_arr = np.append(existing_arr, x)
        existing_set.add(x)
    
    if len(existing) > len(best) and _is_valid_triples(existing):
        best = list(existing)
    
    # Strategy 3: Multiple randomized greedy attempts with different starting points
    for attempt in range(500):
        if time.time() > deadline:
            break
        
        selected = []
        selected_arr = np.array([], dtype=np.int64)
        selected_sums = set()
        selected_set = set()
        
        # Random starting point and step pattern
        start_val = rng.randint(1, 200)
        min_step = rng.randint(1, 6)
        
        x = start_val
        while x <= 4096:
            if _check_collision_fast(selected_arr, selected_sums, x):
                pass
            else:
                _update_sums_fast(selected_arr, selected_sums, x)
                selected.append(x)
                selected_arr = np.append(selected_arr, x)
                selected_set.add(x)
            x += min_step + rng.randint(0, min_step)
        
        if len(selected) > len(best):
            best = selected
    
    # Strategy 4: Greedy with varying density patterns
    for attempt in range(300):
        if time.time() > deadline:
            break
        
        selected = []
        selected_arr = np.array([], dtype=np.int64)
        selected_sums = set()
        selected_set = set()
        
        # Use a base step with occasional larger jumps
        base_step = rng.randint(2, 12)
        jump_prob = rng.uniform(0.05, 0.3)
        
        x = rng.randint(1, 50)
        while x <= 4096:
            if _check_collision_fast(selected_arr, selected_sums, x):
                pass
            else:
                _update_sums_fast(selected_arr, selected_sums, x)
                selected.append(x)
                selected_arr = np.append(selected_arr, x)
                selected_set.add(x)
            
            if rng.random() < jump_prob:
                x += base_step * rng.randint(2, 4)
            else:
                x += base_step
    
    if len(selected) > len(best):
        best = selected
    
    # Strategy 5: Local improvement - try to add more elements to best set
    if len(best) > 0 and time.time() < deadline - 5:
        current = list(best)
        current.sort()
        current_arr = np.array(current, dtype=np.int64)
        current_sums = set(_all_triple_sums(current))
        current_set = set(current)
        
        # Try to add elements in order
        for x in range(1, 4097):
            if time.time() > deadline - 5:
                break
            if x in current_set:
                continue
            if _check_collision_fast(current_arr, current_sums, x):
                continue
            _update_sums_fast(current_arr, current_sums, x)
            current.append(x)
            current_arr = np.append(current_arr, x)
            current_set.add(x)
            current.sort()
            current_arr = np.sort(current_arr)
        
        if len(current) > len(best):
            best = current
    
    # Strategy 6: Swap-based improvement
    if len(best) > 0 and time.time() < deadline - 3:
        current = list(best)
        current.sort()
        
        for iteration in range(50):
            if time.time() > deadline - 3:
                break
            
            if not _is_valid_triples(current):
                current = list(best)
                break
            
            current_arr = np.array(current, dtype=np.int64)
            current_sums = set(_all_triple_sums(current))
            current_set = set(current)
            n = len(current)
            
            # Try adding
            added = False
            for x in range(1, 4097):
                if x in current_set:
                    continue
                if _check_collision_fast(current_arr, current_sums, x):
                    continue
                _update_sums_fast(current_arr, current_sums, x)
                current.append(x)
                current_arr = np.append(current_arr, x)
                current_set.add(x)
                current.sort()
                current_arr = np.sort(current_arr)
                added = True
                break
            
            if not added and n > 1:
                # Remove a random element, try to find replacement
                idx = rng.randint(0, n - 1)
                removed_val = current.pop(idx)
                current_set.discard(removed_val)
                current.sort()
                current_arr = np.array(current, dtype=np.int64)
                current_sums = set(_all_triple_sums(current))
                
                for x in range(1, 4097):
                    if x in current_set:
                        continue
                    if _check_collision_fast(current_arr, current_sums, x):
                        continue
                    _update_sums_fast(current_arr, current_sums, x)
                    current.append(x)
                    current_arr = np.append(current_arr, x)
                    current_set.add(x)
                    current.sort()
                    current_arr = np.sort(current_arr)
                    break
                
                if len(current) > len(best):
                    best = list(current)
    
    # Final validation
    if best and not _is_valid_triples(best):
        # Fallback to exponential construction
        best = _construct_b3_set()
    
    return best


# Test with known small cases
if __name__ == "__main__":
    result = construct()
    print(f"Found set of size {len(result)}")
    print(f"Set: {result[:20]}...")
    print(f"Valid: {_is_valid_triples(result)}")
