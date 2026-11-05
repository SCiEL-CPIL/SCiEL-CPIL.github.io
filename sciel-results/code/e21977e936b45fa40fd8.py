import time
from fractions import Fraction

def construct():
    """
    Finds the subset of denominators [2, 3, ..., 48] that minimizes the deficit 
    1 - sum(1/d) subject to sum(1/d) < 1.
    
    Uses a Branch and Bound DFS approach with exact rational arithmetic.
    Denominators are processed in descending order to maximize early pruning.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill

    denominators = list(range(2, 49))
    n = len(denominators)
    
    # We want to maximize sum(1/d) < 1.
    # Sorting denominators in descending order means we consider larger fractions first.
    # This helps find a high-sum valid subset quickly, which improves the bound for pruning.
    # Index mapping: we need to return indices into the original [2, ..., 48] list.
    # Original index of d is d - 2.
    
    # Create a list of (value, original_index) sorted by value descending
    # value = 1/d
    items = [(Fraction(1, d), d - 2) for d in denominators]
    items.sort(key=lambda x: x[0], reverse=True)
    
    # Precompute suffix sums for pruning
    # suffix[i] = sum of values from items[i] to items[n-1]
    suffix_sums = [Fraction(0, 1)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + items[i][0]
        
    best_sum = Fraction(0, 1)
    best_indices = []
    
    def dfs(idx, current_sum, current_indices):
        nonlocal best_sum, best_indices
        
        if time.time() > deadline:
            return
            
        # Pruning: if current_sum + max possible remaining sum <= best_sum, 
        # we cannot improve, so prune.
        if current_sum + suffix_sums[idx] <= best_sum:
            return
            
        if idx == n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_indices = current_indices[:]
            return
        
        # Option 1: Include items[idx]
        val, orig_idx = items[idx]
        new_sum = current_sum + val
        
        # Constraint: sum must be STRICTLY BELOW 1
        if new_sum < 1:
            current_indices.append(orig_idx)
            dfs(idx + 1, new_sum, current_indices)
            current_indices.pop()
            
        # Option 2: Exclude items[idx]
        dfs(idx + 1, current_sum, current_indices)

    dfs(0, Fraction(0, 1), [])
    
    if not best_indices:
        # Fallback if something went wrong (shouldn't happen as 1/2 is valid)
        return [0]
        
    return best_indices

if __name__ == "__main__":
    res = construct()
    # print(res)
    pass
