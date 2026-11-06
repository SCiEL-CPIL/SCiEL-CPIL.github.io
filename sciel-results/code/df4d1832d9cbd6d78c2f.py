def construct():
    from fractions import Fraction
    import sys
    sys.setrecursionlimit(10000)

    denominators = list(range(2, 49))
    n = len(denominators)
    values = [Fraction(1, d) for d in denominators]
    
    # Sort by value descending to prune faster
    # We need to keep track of original indices
    items = [(values[i], i) for i in range(n)]
    items.sort(key=lambda x: x[0], reverse=True)
    
    # Precompute suffix sums for pruning
    suffix_sums = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + items[i][0]

    best_sum = Fraction(0)
    best_indices = set()
    path = []

    # Branch and Bound
    def dfs(idx, current_sum):
        nonlocal best_sum, best_indices
        
        # Pruning: if we can't beat best_sum even with all remaining items
        # Use float approximation for speed? No, let's stick to Fraction for correctness first.
        # If it's too slow, we can switch.
        if current_sum + suffix_sums[idx] <= best_sum:
            return
            
        # Pruning: if current sum is already >= 1
        if current_sum >= 1:
            return
            
        # Update best if current sum is valid and better
        if current_sum > best_sum and current_sum < 1:
            best_sum = current_sum
            best_indices = set(path)
            
        # If we've considered all items, stop
        if idx == n:
            return
            
        val, orig_idx = items[idx]
        
        # Option 1: Include item
        # Check if including it keeps sum < 1
        new_sum = current_sum + val
        if new_sum < 1:
            path.append(orig_idx)
            dfs(idx + 1, new_sum)
            path.pop()
            
        # Option 2: Exclude item
        dfs(idx + 1, current_sum)

    # Start search
    dfs(0, Fraction(0))
    
    return sorted(best_indices)
