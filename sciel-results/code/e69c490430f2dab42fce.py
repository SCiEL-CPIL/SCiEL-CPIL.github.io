import time
from math import lcm
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    denominators = list(range(2, 49))
    n = len(denominators)

    # Calculate LCM of all denominators
    L = 1
    for d in denominators:
        L = lcm(L, d)

    # We want to maximize sum of 1/d_i such that sum < 1
    # In integer units: maximize sum of (L/d_i) such that sum < L

    weights = [L // d for d in denominators]

    # Sort indices by weight descending (i.e., denominator ascending)
    # But we need to track original indices
    idx_sorted = sorted(range(n), key=lambda i: -weights[i])

    # Suffix sums for pruning: suffix_sum[i] = sum of weights from index i to end in sorted order
    suffix_sum = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[idx_sorted[i]]

    best_sum = -1
    best_mask = 0
    # We'll use a list to track which indices are currently included
    current_included = [False] * n

    def dfs(pos, current_sum, current_mask):
        nonlocal best_sum, best_mask

        if current_sum > best_sum:
            best_sum = current_sum
            best_mask = current_mask

        # Pruning: even if we add all remaining, can we beat best?
        # Max possible = current_sum + suffix_sum[pos]
        # We need current_sum + suffix_sum[pos] > best_sum (strictly, since we need strictly better)
        # Actually we need the final sum to be < L, and we want to maximize it.
        # If current_sum + suffix_sum[pos] <= best_sum, we can't improve
        if current_sum + suffix_sum[pos] <= best_sum:
            return

        if pos >= n:
            return

        # Check time
        if time.time() > deadline:
            return

        idx = idx_sorted[pos]
        w = weights[idx]

        # Option 1: Include this index
        new_sum = current_sum + w
        if new_sum < L:  # Must be strictly below 1
            current_included[idx] = True
            dfs(pos + 1, new_sum, current_mask | (1 << idx))
            current_included[idx] = False

        # Option 2: Skip this index
        dfs(pos + 1, current_sum, current_mask)

    dfs(0, 0, 0)

    if best_mask == 0:
        return [0]

    result_indices = []
    for i in range(n):
        if best_mask & (1 << i):
            result_indices.append(i)

    return result_indices
