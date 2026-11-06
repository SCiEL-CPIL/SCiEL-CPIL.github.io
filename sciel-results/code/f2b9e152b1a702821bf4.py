def construct():
    from fractions import Fraction
    import bisect

    denominators = list(range(2, 49))
    n = len(denominators)
    values = [Fraction(1, d) for d in denominators]
    
    # Split into two halves for Meet-in-the-Middle
    mid = n // 2
    left_indices = list(range(mid))
    right_indices = list(range(mid, n))
    
    # Generate all subset sums <= 1 for a list of indices
    def gen_sums(indices):
        sums = []
        
        def dfs(start, current_sum, current_mask):
            if current_sum > 1:
                return
            if start == len(indices):
                sums.append((current_sum, current_mask))
                return
            # Option 1: Don't take index 'start'
            dfs(start + 1, current_sum, current_mask)
            # Option 2: Take index 'start'
            idx_val = indices[start]
            new_sum = current_sum + values[idx_val]
            if new_sum <= 1:
                dfs(start + 1, new_sum, current_mask | (1 << start))
            
        dfs(0, Fraction(0), 0)
        return sums

    left_sums = gen_sums(left_indices)
    right_sums = gen_sums(right_indices)
    
    # Sort left_sums by sum to enable binary search
    left_sums.sort(key=lambda x: x[0])
    left_sum_values = [s[0] for s in left_sums]
    
    best_deficit = Fraction(1) # Initial best is deficit 1 (sum 0)
    best_left_mask = 0
    best_right_mask = 0
    
    # Iterate through right sums and find best compatible left sum
    for s2, m2 in right_sums:
        target = Fraction(1) - s2
        # We want largest s1 such that s1 < target
        # bisect_left returns the first index where left_sum_values[pos] >= target
        pos = bisect.bisect_left(left_sum_values, target)
        if pos > 0:
            idx = pos - 1
            s1, m1 = left_sums[idx]
            total_sum = s1 + s2
            # Check strictly below 1
            if total_sum < 1:
                deficit = Fraction(1) - total_sum
                if deficit < best_deficit:
                    best_deficit = deficit
                    best_left_mask = m1
                    best_right_mask = m2
        
    # Reconstruct indices
    result_indices = []
    # Left part
    for i in range(len(left_indices)):
        if best_left_mask & (1 << i):
            result_indices.append(left_indices[i])
    # Right part
    for i in range(len(right_indices)):
        if best_right_mask & (1 << i):
            result_indices.append(right_indices[i])
            
    return sorted(result_indices)
