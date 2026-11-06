import time
from fractions import Fraction

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Items: (value_fraction, original_index)
    items = [(Fraction(1, d), i) for i, d in enumerate(denominators)]
    # Sort by value descending
    items.sort(key=lambda x: -x[0])
    
    best_sum = Fraction(0)
    best_indices = []
    
    # Precompute suffix sums
    suffix_sums = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + items[i][0]
        
    start_time = time.time()
    deadline = start_time + 50.0
    
    def bnb(idx, current_sum, current_indices):
        nonlocal best_sum, best_indices
        
        if time.time() > deadline:
            return
            
        # Upper bound
        ub = current_sum + suffix_sums[idx]
        if ub <= best_sum:
            return
            
        if idx == n:
            if current_sum < 1 and current_sum > best_sum:
                best_sum = current_sum
                best_indices = list(current_indices)
            return
        
        val, orig_idx = items[idx]
        
        # Option 1: Include item
        new_sum = current_sum + val
        if new_sum < 1:
            current_indices.append(orig_idx)
            bnb(idx + 1, new_sum, current_indices)
            current_indices.pop()
            
        # Option 2: Exclude item
        bnb(idx + 1, current_sum, current_indices)

    bnb(0, Fraction(0), [])
    
    if not best_indices:
        return [0]
        
    return best_indices
