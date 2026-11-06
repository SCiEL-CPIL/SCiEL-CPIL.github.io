"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    from fractions import Fraction
    import sys
    
    # Denominators are 2 to 48
    # We want indices into [2, 3, ..., 48]
    # Let's work with denominators directly and map back to indices
    
    # Directive: iterate in descending order of denominators
    # So we process 48, 47, ..., 2
    denoms = list(range(48, 1, -1)) # [48, 47, ..., 2]
    n = len(denoms)
    
    # Precompute suffix sums of reciprocals for pruning
    # suffix_sum[i] = sum(1/denoms[j] for j in range(i, n))
    suffix_sum = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + Fraction(1, denoms[i])
    
    best_sum = Fraction(0)
    best_indices = []
    
    # Initialize with a good heuristic solution to tighten the bound
    # Greedy: Pick largest fractions that fit
    # This is equivalent to trying to make the sum close to 1
    heuristic_indices = []
    current_sum = Fraction(0)
    # We want to maximize the sum < 1.
    # A common heuristic is to pick small denominators (large values) first.
    # Let's try a simple greedy on ascending denominators (2, 3, ...)
    for d in range(2, 49):
        if current_sum + Fraction(1, d) < 1:
            current_sum += Fraction(1, d)
            heuristic_indices.append(d - 2)
            
    # Refine heuristic: Try to swap or add?
    # The simple greedy gave {2,3,7} -> 41/42.
    # We can manually add the best next term if possible.
    # Deficit is 1/42. We need 1/d < 1/42 => d > 42.
    # d=43 is available.
    if current_sum + Fraction(1, 43) < 1:
         current_sum += Fraction(1, 43)
         heuristic_indices.append(43 - 2)
         
    if current_sum > best_sum:
        best_sum = current_sum
        best_indices = heuristic_indices[:]

    # Branch and Bound
    # State: idx, current_sum, current_indices
    def dfs(idx, current_sum, current_indices):
        nonlocal best_sum, best_indices
        
        # Pruning 1: If current sum is >= 1, invalid
        if current_sum >= 1:
            return
            
        # Pruning 2: If even adding all remaining reciprocals doesn't beat best_sum
        if current_sum + suffix_sum[idx] <= best_sum:
            return
            
        # Base case
        if idx == n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_indices = current_indices[:]
            return
        
        d = denoms[idx]
        
        # Option 1: Skip this denominator
        dfs(idx + 1, current_sum, current_indices)
        
        # Option 2: Take this denominator
        new_sum = current_sum + Fraction(1, d)
        if new_sum < 1:
            current_indices.append(idx)
            dfs(idx + 1, new_sum, current_indices)
            current_indices.pop()

    dfs(0, Fraction(0), [])
    
    # Map internal indices (in descending list) back to original indices
    # denoms[j] is the denominator at position j in the descending list.
    # The original index for denominator D is D - 2.
    # Wait, the `current_indices` in dfs stores the index `idx` into the `denoms` list.
    # We need to return indices into [2, 3, ..., 48].
    # So for each `i` in `best_indices`, the denominator is `denoms[i]`.
    # The required index is `denoms[i] - 2`.
    
    final_indices = [denoms[i] - 2 for i in best_indices]
    
    # Ensure non-empty
    if not final_indices:
        # Fallback, though best_sum should be > 0
        return [0]
        
    return final_indices
# EVOLVE-BLOCK-END
