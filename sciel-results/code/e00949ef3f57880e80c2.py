"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import sys
import math
import time
import bisect
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s deadline for safety (60s limit)

    denominators = list(range(2, 49))  # [2, 3, ..., 48]
    n = len(denominators)

    # Calculate LCM of 2..48 for exact integer arithmetic
    L = 1
    for d in denominators:
        L = L * d // math.gcd(L, d)

    # Weights: w[i] = L / denominators[i]
    weights = [L // d for d in denominators]
    
    # Create list of (weight, index) and sort by weight descending
    # This helps find large sums early, improving the lower bound for pruning
    items = [(weights[i], i) for i in range(n)]
    items.sort(key=lambda x: -x[0])
    
    # Re-ordered weights and indices
    sorted_weights = [w for w, _ in items]
    sorted_indices = [i for _, i in items]
    
    # Suffix sums: suffix_sum[i] is the sum of sorted_weights[i:]
    suffix_sum = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + sorted_weights[i]

    best_sum = 0
    best_mask = 0  # Bitmask of selected items in the sorted order

    # Recursive B&B
    # i: current index in sorted_weights
    # current_sum: sum of selected weights so far
    # current_mask: bitmask of selected items so far
    sys.setrecursionlimit(10000)

    def dfs(i, current_sum, current_mask):
        nonlocal best_sum, best_mask
        
        if time.time() > deadline:
            return

        if i == n:
            if current_sum > best_sum and current_sum < L:
                best_sum = current_sum
                best_mask = current_mask
            return

        # Pruning 1: If current_sum + sum of all remaining <= best_sum, no improvement possible
        if current_sum + suffix_sum[i] <= best_sum:
            return
            
        # Pruning 2: If current_sum >= L, we can't add more (since weights are positive)
        # But we must strictly be < L. If current_sum == L, it's invalid.
        # Since we want max sum < L, if current_sum >= L, we prune.
        if current_sum >= L:
            return

        # Option 1: Include item i
        new_sum = current_sum + sorted_weights[i]
        if new_sum < L: # Only valid if strictly less than L
            # Check if this new sum is better than current best
            # We don't update best here immediately unless we reach leaf or if we want to update bound early?
            # Updating best early helps pruning in other branches.
            if new_sum > best_sum:
                best_sum = new_sum
                best_mask = current_mask | (1 << i)
            dfs(i + 1, new_sum, current_mask | (1 << i))
            
        # Option 2: Exclude item i
        # Even if we exclude, we might still find a better sum by combining later items?
        # Yes, so we must recurse.
        dfs(i + 1, current_sum, current_mask)

    dfs(0, 0, 0)

    # Reconstruct indices from best_mask (which is in sorted order)
    indices = []
    for i in range(n):
        if best_mask & (1 << i):
            indices.append(sorted_indices[i])
            
    # Ensure non-empty
    if not indices:
        # Fallback: pick the smallest denominator (largest fraction) that is < 1?
        # 1/2 = 0.5 < 1. So index 0 (denom 2) is always valid.
        indices = [0]

    # Verify with exact arithmetic
    unit_fracs = [Fraction(1, d) for d in denominators]
    total = sum(unit_fracs[i] for i in indices)
    if total >= 1:
        # Fallback logic: remove largest element
        indices.sort(key=lambda i: -unit_fracs[i])
        indices = indices[:-1]
        if not indices:
            indices = [0]

    return sorted(indices)
# EVOLVE-BLOCK-END
