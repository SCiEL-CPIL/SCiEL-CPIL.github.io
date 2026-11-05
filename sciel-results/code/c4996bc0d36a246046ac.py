import time
from fractions import Fraction
import sys

def construct():
    start = time.time()
    deadline = start + 85.0  # Safety margin under the 90s wall-clock limit

    denominators = list(range(2, 49))
    n = len(denominators)

    # Precompute suffix sums of reciprocals for pruning
    # suffix_sum[i] = sum of 1/denominators[j] for j from i to n-1
    suffix_sum = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + 1.0 / denominators[i]

    best_sum_frac = Fraction(0)
    best_sum_float = 0.0
    best_mask = 0

    # Initial Greedy to get a strong lower bound on the sum (upper bound on deficit)
    # We try to pack as much as possible.
    # Strategy: Iterate through denominators. If adding it keeps sum < 1, add it.
    # This is a simple heuristic.
    curr_sum = 0.0
    curr_mask = 0
    for i in range(n):
        r = 1.0 / denominators[i]
        if curr_sum + r < 1.0:
            curr_sum += r
            curr_mask |= (1 << i)
    
    # Verify and update
    if curr_mask:
        s_frac = sum(Fraction(1, denominators[i]) for i in range(n) if curr_mask & (1 << i))
        if s_frac < 1 and s_frac > best_sum_frac:
            best_sum_frac = s_frac
            best_sum_float = float(s_frac)
            best_mask = curr_mask

    # Branch and Bound
    sys.setrecursionlimit(10000)

    def dfs(i, curr_sum_float, mask):
        nonlocal best_sum_frac, best_sum_float, best_mask

        # Time check
        if time.time() > deadline:
            return

        # Prune 1: Exceeded 1 (with small epsilon for float safety, but we verify at leaf)
        # If curr_sum_float >= 1.0, we can't add more.
        if curr_sum_float >= 1.0:
            return

        # Prune 2: Even taking all remaining, we can't beat the best found
        if curr_sum_float + suffix_sum[i] <= best_sum_float:
            return

        # Base case
        if i == n:
            # Verify exact sum using Fraction
            s_frac = sum(Fraction(1, denominators[j]) for j in range(n) if mask & (1 << j))
            if s_frac < 1 and s_frac > best_sum_frac:
                best_sum_frac = s_frac
                best_sum_float = float(s_frac)
                best_mask = mask
            return

        # Option 1: Take item i
        # Check if adding it keeps sum < 1 (approximate check for speed)
        r_i = 1.0 / denominators[i]
        if curr_sum_float + r_i < 1.0:
            dfs(i + 1, curr_sum_float + r_i, mask | (1 << i))

        # Option 2: Skip item i
        dfs(i + 1, curr_sum_float, mask)

    dfs(0, 0.0, 0)

    # Extract selected indices
    selected = [i for i in range(n) if best_mask & (1 << i)]

    # Final verification
    if not selected:
        # Fallback: 1/2 is always valid
        return [0]

    s_frac = sum(Fraction(1, denominators[i]) for i in selected)
    if s_frac >= 1:
        # This is a safety net. If it happens, remove the element that contributes most to the excess
        # Or just return a known valid subset
        # Let's find a valid subset by removing elements until sum < 1
        # Since we minimized deficit, removing the largest reciprocal is best?
        # Actually, just remove the last added one?
        # Simple fallback: return [0] (1/2) if verification fails
        return [0]

    return selected
