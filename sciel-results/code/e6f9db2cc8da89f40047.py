def construct():
    from fractions import Fraction
    import sys

    sys.setrecursionlimit(10000)

    denominators = list(range(2, 49))
    n = len(denominators)
    vals = [Fraction(1, d) for d in denominators]
    
    # Sort by value descending
    indices = list(range(n))
    indices.sort(key=lambda i: vals[i], reverse=True)
    sorted_vals = [vals[i] for i in indices]
    
    # Precompute suffix sums
    suffix_sums = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + sorted_vals[i]
        
    best_sum = Fraction(0)
    best_indices = []
    
    # Greedy initial solution
    current_sum = Fraction(0)
    greedy_indices = []
    for i in range(n):
        if current_sum + sorted_vals[i] < 1:
            current_sum += sorted_vals[i]
            greedy_indices.append(i)
            
    if current_sum > best_sum:
        best_sum = current_sum
        best_indices = greedy_indices[:]

    # DFS Branch and Bound
    def dfs(pos, current_sum, current_indices):
        nonlocal best_sum, best_indices
        
        # Pruning: Max possible sum from here is current_sum + suffix_sums[pos]
        if current_sum + suffix_sums[pos] <= best_sum:
            return
            
        if pos == n:
            return
            
        # Option 1: Include
        new_sum = current_sum + sorted_vals[pos]
        if new_sum < 1:
            if new_sum > best_sum:
                best_sum = new_sum
                best_indices = current_indices + [pos]
            dfs(pos + 1, new_sum, current_indices + [pos])
            
        # Option 2: Exclude
        dfs(pos + 1, current_sum, current_indices)

    dfs(0, Fraction(0), [])
    
    # Map back to original indices
    original_best_indices = [indices[i] for i in best_indices]
    
    return original_best_indices
