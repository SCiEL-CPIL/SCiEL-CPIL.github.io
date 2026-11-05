import time
from fractions import Fraction
import heapq

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin under the 90s wall-clock limit

    denominators = list(range(2, 49))
    n = len(denominators)

    # Precompute float weights for fast comparison
    weights = [1.0 / d for d in denominators]
    
    # We want to find a subset S such that sum_{i in S} 1/d_i < 1, 
    # minimizing 1 - sum_{i in S} 1/d_i.
    # This is equivalent to maximizing sum_{i in S} 1/d_i subject to sum < 1.
    
    # Since we need the sum to be STRICTLY below 1, and the values are rational,
    # the maximum possible sum below 1 is achieved by the subset with the largest sum < 1.
    
    # A simple greedy approach often works well for knapsack-like problems, 
    # but it doesn't guarantee optimality.
    # However, given the structure (unit fractions), the optimal subset is likely 
    # composed of small denominators (large weights) but carefully chosen to stay below 1.
    
    # Let's try a Branch and Bound approach using floats for speed, 
    # then verify the top candidates with exact arithmetic.
    
    # Sort by weight descending (i.e., denominator ascending)
    # indices sorted by denominator
    order = sorted(range(n), key=lambda i: denominators[i])
    
    # We will explore subsets in a DFS manner, pruning branches that cannot beat the current best.
    # State: (index, current_sum_float, current_mask)
    # We want to maximize current_sum_float < 1.
    
    best_sum_float = 0.0
    best_mask = 0
    best_deficit_exact = None
    
    # To make B&B effective, we need an upper bound on the potential sum from remaining items.
    # Since items are sorted by weight descending, the sum of remaining weights is an upper bound.
    # Let suffix_sum[i] = sum of weights from index i to end.
    suffix_sum = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[order[i]]
    
    # DFS
    # We use a stack to avoid recursion depth issues, or just recursion if n is small (n=47).
    # Recursion depth 47 is fine.
    
    def dfs(idx, current_sum, mask):
        nonlocal best_sum_float, best_mask, best_deficit_exact
        
        # Pruning: if current_sum + suffix_sum[idx] <= best_sum_float, we can't improve
        if current_sum + suffix_sum[idx] <= best_sum_float:
            return
        
        # Check time
        if time.time() > deadline - 2.0:
            return
        
        if idx == n:
            if current_sum < 1.0 and current_sum > best_sum_float:
                best_sum_float = current_sum
                best_mask = mask
            return
        
        # Option 1: Skip item idx
        dfs(idx + 1, current_sum, mask)
        
        # Option 2: Take item idx
        new_sum = current_sum + weights[order[idx]]
        if new_sum < 1.0:
            new_mask = mask | (1 << order[idx])
            dfs(idx + 1, new_sum, new_mask)
        # If new_sum >= 1.0, we cannot take it (since we need strict < 1)
        # And we cannot take any further items in this branch if we were to add them? 
        # No, we just skip this item. But wait, if we skip this item, we already handled that above.
        # If we take it and it exceeds 1, that branch is invalid.
        
    # The above DFS might be too slow if it explores too many branches.
    # Let's add a heuristic: if we are close to the deadline, stop.
    # Also, the "suffix_sum" bound is loose because it assumes we can take all remaining items,
    # but the constraint sum < 1 might prevent taking all.
    # A tighter bound: max possible sum from remaining is min(suffix_sum[idx], 1.0 - current_sum - epsilon)
    # But since we are maximizing, if current_sum + suffix_sum[idx] <= best_sum_float, prune.
    
    # Let's run the DFS
    dfs(0, 0.0, 0)
    
    # Now we have a candidate best_mask. Let's verify it exactly.
    if best_mask != 0:
        selected = [i for i in range(n) if best_mask & (1 << i)]
        exact_sum = sum(Fraction(1, denominators[i]) for i in selected)
        if exact_sum < 1:
            return selected
        # If exact_sum >= 1 (due to float error), we need to find a valid subset.
        # This is unlikely if float arithmetic is careful, but possible.
        # In that case, we can try removing the smallest item from the set.
        # Or just fall back to a greedy construction.
    
    # Fallback: Greedy construction
    # Pick items one by one in order of largest weight (smallest denominator)
    # if they keep the sum strictly below 1.
    selected = []
    s = Fraction(0)
    for k in order:
        if s + Fraction(1, denominators[k]) < 1:
            selected.append(k)
            s += Fraction(1, denominators[k])
    
    if not selected:
        # Should not happen, but if it does, pick the largest denominator
        selected = [n - 1]
        
    return selected

