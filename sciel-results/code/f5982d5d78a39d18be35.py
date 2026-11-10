# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import sys
    sys.setrecursionlimit(10000)

    denominators = list(range(2, 49))
    n = len(denominators)
    values = [Fraction(1, d) for d in denominators]
    
    # Sort by value descending to improve branch and bound efficiency
    # We keep track of original indices
    indexed_values = sorted([(v, i) for i, v in enumerate(values)], reverse=True)
    sorted_values = [v for v, i in indexed_values]
    original_indices = [i for v, i in indexed_values]
    
    # Precompute suffix sums
    suffix_sums = [Fraction(0) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + sorted_values[i]
        
    # Initialize with a greedy solution to provide a strong lower bound for pruning
    best_sum = Fraction(0)
    best_indices = []
    
    current_sum = Fraction(0)
    current_indices = []
    for i in range(n):
        if current_sum + sorted_values[i] < 1:
            current_sum += sorted_values[i]
            current_indices.append(original_indices[i])
            
    best_sum = current_sum
    best_indices = current_indices
    
    def dfs(idx, curr_sum, curr_idx):
        nonlocal best_sum, best_indices
        
        # Pruning: If we can't beat the best sum even by taking all remaining, stop
        if curr_sum + suffix_sums[idx] <= best_sum:
            return
            
        # If we've considered all items
        if idx == n:
            if curr_sum > best_sum:
                best_sum = curr_sum
                best_indices = curr_idx[:]
            return
        
        # Option 1: Include sorted_values[idx]
        # Only if it doesn't violate the strict < 1 constraint
        new_sum = curr_sum + sorted_values[idx]
        if new_sum < 1:
            curr_idx.append(original_indices[idx])
            dfs(idx + 1, new_sum, curr_idx)
            curr_idx.pop()
            
        # Option 2: Exclude sorted_values[idx]
        dfs(idx + 1, curr_sum, curr_idx)
        
    dfs(0, Fraction(0), [])
    
    return best_indices
# EVOLVE-BLOCK-END
