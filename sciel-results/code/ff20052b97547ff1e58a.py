"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from fractions import Fraction
import sys

def construct():
    n = 256
    
    # 1. Build conflict graph
    # Node i represents number i+1
    # Conflict if they form a 3-AP with any other node in [1, n]
    conflict_masks = [0] * n
    for i in range(n):
        u = i + 1
        for j in range(n):
            if i == j: continue
            v = j + 1
            # Check if u, v form a 3-AP with some w in [1, n]
            # Case 1: u is midpoint. v, u, w -> w = 2u - v
            w1 = 2 * u - v
            if 1 <= w1 <= n and w1 != v and w1 != u:
                conflict_masks[i] |= (1 << j)
                continue
            # Case 2: v is midpoint. u, v, w -> w = 2v - u
            w2 = 2 * v - u
            if 1 <= w2 <= n and w2 != u and w2 != v:
                conflict_masks[i] |= (1 << j)
                continue
            # Case 3: u, v are endpoints. w = (u+v)/2
            if (u + v) % 2 == 0:
                w3 = (u + v) // 2
                if 1 <= w3 <= n and w3 != u and w3 != v:
                    conflict_masks[i] |= (1 << j)
                    continue

    # 2. Weights
    weights = [1.0 / (i + 1) for i in range(n)]
    
    # 3. Initial Greedy Solution for Lower Bound
    # Greedy: pick smallest available (highest weight) that doesn't conflict with selected
    # This gives a valid independent set.
    initial_selected = []
    available = (1 << n) - 1
    # We process in order of weight (which is index 0 to n-1)
    # But we need to ensure we pick a valid set.
    # A simple greedy: iterate i from 0 to n-1. If i is available, select it, remove i and its conflicts.
    # This maximizes the number of small indices? Not necessarily optimal sum, but good start.
    for i in range(n):
        if (available >> i) & 1:
            initial_selected.append(i + 1)
            available &= ~conflict_masks[i]
            available &= ~(1 << i)
            
    initial_sum = sum(1.0 / x for x in initial_selected)
    
    best_sum = initial_sum
    best_set = initial_selected[:]
    
    # 4. Branch and Bound
    # To speed up, we can use a recursive function.
    # State: candidates_mask, current_sum, current_set (list of indices)
    
    sys.setrecursionlimit(10000)
    
    def get_bit_count(mask):
        return bin(mask).count("1")
        
    def search(candidates_mask, current_sum, current_set):
        nonlocal best_sum, best_set
        
        # Pruning: Upper bound is current_sum + sum of weights of all candidates
        # This is a valid upper bound.
        # Optimization: Precompute sum of weights for masks? Too many masks.
        # Calculate sum of weights for candidates.
        # We can iterate bits.
        upper_bound = current_sum
        m = candidates_mask
        while m:
            lsb = m & -m
            idx = lsb.bit_length() - 1
            upper_bound += weights[idx]
            m -= lsb
            
        if upper_bound <= best_sum:
            return
            
        # If no candidates, update best
        if candidates_mask == 0:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = current_set[:]
            return
            
        # Select a node to branch on
        # Heuristic: Pick the node with the largest weight among candidates?
        # Or the one with the most conflicts?
        # Let's pick the node with the largest weight (smallest index) to prioritize high values.
        # Find smallest index in candidates_mask
        lsb = candidates_mask & -candidates_mask
        u_idx = lsb.bit_length() - 1
        
        # Branch 1: Include u_idx
        # Remove u_idx and its conflicts from candidates
        new_candidates_inc = candidates_mask & ~(1 << u_idx) & ~conflict_masks[u_idx]
        search(new_candidates_inc, current_sum + weights[u_idx], current_set + [u_idx + 1])
        
        # Branch 2: Exclude u_idx
        new_candidates_exc = candidates_mask & ~(1 << u_idx)
        search(new_candidates_exc, current_sum, current_set)
        
    # Start search
    # We can start with all nodes available.
    # However, the greedy already picked a set. 
    # To be safe and fast, we can just run B&B on the full set.
    # The initial lower bound helps prune.
    
    search((1 << n) - 1, 0.0, [])
    
    return best_set

# EVOLVE-BLOCK-END
