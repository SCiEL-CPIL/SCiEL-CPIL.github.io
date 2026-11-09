"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import heapq
from fractions import Fraction
from math import lcm

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    fracs = [Fraction(1, d) for d in denominators]
    
    # Compute LCM of all denominators for exact integer arithmetic
    L = 1
    for d in denominators:
        L = lcm(L, d)
    
    # Integer weights: w_i = L / d_i, so sum of selected weights / L = reciprocal sum
    weights = [L // d for d in denominators]
    total_weight = sum(weights)
    
    # We want to maximize sum of selected weights subject to sum < L
    # Equivalently, minimize deficit = L - sum, where deficit > 0
    
    # Dijkstra on deficit space: state = (deficit, frozenset of used indices)
    # But that's too large. Instead, use A* / Dijkstra with pruning.
    
    # Better approach: Dijkstra where state is just the current sum (as integer),
    # and we track which indices have been used via a bitset.
    # Since n=47, we can use a 47-bit integer as the mask.
    
    # Start: sum=0, mask=0, deficit=L
    # Goal: find minimum deficit > 0 where we can add more items without exceeding L
    
    # Actually, let's think differently. We want subset S such that sum(S) < L and 
    # L - sum(S) is minimized. This is equivalent to finding the largest subset sum < L.
    
    # Dijkstra: state = (current_sum, mask). Edge: add item i if not in mask and new_sum < L.
    # But this is still exponential in the worst case.
    
    # Key insight: We can use Dijkstra where the "distance" is the deficit, and we 
    # explore states in order of increasing deficit. The first valid state we find 
    # that can be extended (or is terminal) gives us the answer.
    
    # Alternative: Meet-in-the-middle or branch and bound with strong pruning.
    
    # Let's try a different structural approach: 
    # Sort items by weight descending. Use branch and bound with:
    # - Upper bound: current sum + sum of all remaining items
    # - Lower bound on deficit: L - upper_bound
    # - If lower_bound >= current_best_deficit, prune
    
    start_time = time.time()
    timeout = 55.0
    
    # Sort by weight descending for better pruning
    order = sorted(range(n), key=lambda i: weights[i], reverse=True)
    sorted_weights = [weights[i] for i in order]
    sorted_fracs = [fracs[i] for i in order]
    
    # Suffix sums for pruning
    suffix_sums = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + sorted_weights[i]
    
    # Greedy initial solution
    best_deficit = L  # Initial: empty set has deficit L
    best_mask = 0
    current_sum = 0
    for i in range(n):
        if current_sum + sorted_weights[i] < L:
            current_sum += sorted_weights[i]
            best_mask |= (1 << i)
    best_deficit = L - current_sum
    
    # Branch and bound DFS
    def dfs(i, current_sum, current_mask):
        nonlocal best_deficit, best_mask, start_time, timeout
        
        if time.time() - start_time > timeout:
            return
        
        # Update best
        if current_sum > 0:
            deficit = L - current_sum
            if deficit < best_deficit:
                best_deficit = deficit
                best_mask = current_mask
        
        if i == n:
            return
        
        # Prune: even if we add all remaining, can we beat best?
        max_possible = current_sum + suffix_sums[i]
        if max_possible <= 0:
            return
        min_possible_deficit = L - max_possible
        if min_possible_deficit >= best_deficit:
            return
        
        # Option 1: Include item i
        new_sum = current_sum + sorted_weights[i]
        if new_sum < L:
            dfs(i + 1, new_sum, current_mask | (1 << i))
        
        # Option 2: Exclude item i
        dfs(i + 1, current_sum, current_mask)
    
    dfs(0, 0, 0)
    
    # Convert best_mask back to original indices
    best_indices = []
    for i in range(n):
        if best_mask & (1 << i):
            best_indices.append(order[i])
    
    return sorted(best_indices)
# EVOLVE-BLOCK-END
