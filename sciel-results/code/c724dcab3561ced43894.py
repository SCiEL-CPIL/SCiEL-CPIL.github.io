def construct():
    from fractions import Fraction
    import time
    
    start_time = time.time()
    deadline = 85.0 # Safety margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Sort by reciprocal value descending (denominator ascending)
    # Index 0 corresponds to denom 2, Index 1 to denom 3, etc.
    # We need to map back to original indices.
    # Original indices: denom 2 is index 0, denom 3 is index 1, ... denom 48 is index 46.
    # So if we sort by value desc, the order is already 2, 3, ..., 48.
    # So sorted_idx[i] = i.
    
    recip = [Fraction(1, d) for d in denominators]
    
    # Suffix sums for pruning
    # suffix_sum[i] = sum(recip[i] + ... + recip[n-1])
    suffix_sum = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + recip[i]
    
    best_sum = Fraction(0)
    best_indices = []
    
    # To speed up, we can use a list for current indices and backtrack
    current_indices = [0] * n
    
    def search(idx, curr_sum, depth):
        nonlocal best_sum, best_indices
        
        # Check time
        if time.time() > deadline:
            return False # Signal to stop
        
        # Prune if current sum is already >= 1
        if curr_sum >= 1:
            return True
            
        # Prune if upper bound cannot improve best
        # Upper bound is curr_sum + sum of all remaining items
        upper_bound = curr_sum + suffix_sum[idx]
        if upper_bound <= best_sum:
            return True
            
        # If we reached the end
        if idx == n:
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_indices = current_indices[:depth]
            return True
        
        # Option 1: Include item idx
        # If adding it keeps sum < 1
        new_sum = curr_sum + recip[idx]
        if new_sum < 1:
            current_indices[depth] = idx # Store original index (which is same as sorted index here)
            search(idx + 1, new_sum, depth + 1)
            if time.time() > deadline:
                return False
        
        # Option 2: Exclude item idx
        search(idx + 1, curr_sum, depth)
        
        return True

    # Start search
    # We want to find the best sum < 1.
    # Initial best is 0 (empty set).
    # However, empty set sum is 0. We need non-empty? 
    # "Return nonempty distinct zero-based indices".
    # If best_indices is empty at the end, we might need to handle that, 
    # but since 1/2 < 1, we will always find at least one.
    
    search(0, Fraction(0), 0)
    
    if not best_indices:
        # Fallback, though unlikely
        return [0]
        
    return best_indices
