"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)
    # Values: 1/d for d in denominators
    values = [Fraction(1, d) for d in denominators]
    
    # Sort by value descending (i.e., by denominator ascending)
    # denominators are already in ascending order, so values are descending
    # indices: 0->2, 1->3, ..., 46->48
    
    best_sum = Fraction(0)
    best_indices = []
    
    # Precompute suffix sums for upper bound
    suffix_sum = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + values[i]
    
    def branch_and_bound(idx, current_sum, selected):
        nonlocal best_sum, best_indices
        nonlocal deadline
        
        if time.time() > deadline:
            return
        
        # Upper bound: current_sum + suffix_sum[idx]
        ub = current_sum + suffix_sum[idx]
        if ub <= best_sum:
            return
        
        if idx == n:
            if current_sum < 1 and current_sum > best_sum:
                best_sum = current_sum
                best_indices = list(selected)
            return
        
        # Try including item idx
        new_sum = current_sum + values[idx]
        if new_sum < 1:
            selected.append(idx)
            branch_and_bound(idx + 1, new_sum, selected)
            selected.pop()
        
        # Try excluding item idx
        branch_and_bound(idx + 1, current_sum, selected)
    
    branch_and_bound(0, Fraction(0), [])
    
    if not best_indices:
        # Fallback: greedy
        selected, total = [], Fraction()
        for i, d in enumerate(denominators):
            if total + Fraction(1, d) < 1:
                selected.append(i)
                total += Fraction(1, d)
        best_indices = selected
    
    return best_indices
# EVOLVE-BLOCK-END
