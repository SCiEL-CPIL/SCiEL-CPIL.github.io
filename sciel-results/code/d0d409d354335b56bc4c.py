def construct():
    from fractions import Fraction
    import time

    start_time = time.time()
    deadline = 50.0 # Safety margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Create list of (value, original_index)
    items = [(Fraction(1, d), i) for i, d in enumerate(denominators)]
    
    # Sort by value descending to improve pruning (larger items first)
    # This makes suffix sums (max possible addition) tighter earlier?
    # Actually, if we sort descending, the suffix sum at index i is the sum of the 
    # remaining smaller items. 
    # Wait, standard B&B for subset sum often sorts descending so that 
    # if we include a large item, we fill the capacity fast.
    # Here capacity is 1.
    items.sort(key=lambda x: x[0], reverse=True)
    
    # Precompute suffix sums
    # suffix_sum[i] = sum of values from i to n-1
    suffix_sum = [Fraction(0) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + items[i][0]
    
    best_sum = Fraction(0)
    best_indices = []
    
    # Initialize with greedy solution to have a good bound
    greedy_sum = Fraction(0)
    greedy_indices = []
    for val, idx in items:
        if greedy_sum + val < 1:
            greedy_sum += val
            greedy_indices.append(idx)
    if greedy_sum > best_sum:
        best_sum = greedy_sum
        best_indices = greedy_indices[:]

    def solve(idx, current_sum, current_indices):
        nonlocal best_sum, best_indices
        
        # Time check
        if time.time() > deadline:
            return

        # Pruning: if current_sum + max_possible < best_sum, no need to continue
        if current_sum + suffix_sum[idx] <= best_sum:
            return
            
        # If we've considered all items
        if idx == n:
            return

        # Option 1: Include item idx
        val, orig_idx = items[idx]
        new_sum = current_sum + val
        
        if new_sum < 1:
            if new_sum > best_sum:
                best_sum = new_sum
                best_indices = current_indices + [orig_idx]
            
            # Recurse with item included
            solve(idx + 1, new_sum, best_indices)
            
        # Option 2: Exclude item idx
        # Only recurse if it's possible to beat best_sum
        # If current_sum + suffix_sum[idx+1] > best_sum
        if current_sum + suffix_sum[idx + 1] > best_sum:
            solve(idx + 1, current_sum, current_indices)

    solve(0, Fraction(0), [])
    
    return best_indices
