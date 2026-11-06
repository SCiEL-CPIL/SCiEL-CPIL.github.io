import time
import math
import bisect
from fractions import Fraction

def construct():
    denominators = list(range(2, 49))
    N = len(denominators)
    
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    L = 1
    for d in denominators:
        L = (L * d) // gcd(L, d)
        
    weights = [L // d for d in denominators]
    
    # Greedy Baseline
    indices_sorted = sorted(range(N), key=lambda i: denominators[i])
    greedy_indices = []
    greedy_sum = 0
    for i in indices_sorted:
        if greedy_sum + weights[i] < L:
            greedy_indices.append(i)
            greedy_sum += weights[i]
    
    best_sum = greedy_sum
    best_indices = greedy_indices
    
    start_time = time.time()
    time_limit = 55.0 # Safety margin for 60s limit

    # Sort indices by weight descending for Branch and Bound
    # Smaller denominators have larger weights
    sorted_indices = sorted(range(N), key=lambda i: weights[i], reverse=True)
    sorted_weights = [weights[i] for i in sorted_indices]
    
    # Precompute suffix sums for pruning
    # suffix_sum[i] = sum of sorted_weights[i:]
    suffix_sum = [0] * (N + 1)
    for i in range(N - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + sorted_weights[i]
    
    # Branch and Bound
    # We use a stack to simulate recursion to avoid recursion limit and overhead
    # State: (idx, current_sum, mask)
    # mask is a bitmask of selected indices in the sorted order
    
    best_sum_bb = best_sum
    best_mask_bb = 0
    
    # Initialize mask for greedy? No, just start B&B from scratch but use best_sum for pruning.
    # We need to map mask back to original indices.
    
    stack = [(0, 0, 0)]
    
    while stack:
        if time.time() - start_time > time_limit:
            break
            
        idx, c_sum, mask = stack.pop()
        
        if idx == N:
            if c_sum > best_sum_bb:
                best_sum_bb = c_sum
                best_mask_bb = mask
            continue
            
        # Pruning: if current sum + remaining sum <= best found, no point
        if c_sum + suffix_sum[idx] <= best_sum_bb:
            continue
            
        # Option 1: Exclude item idx
        stack.append((idx + 1, c_sum, mask))
        
        # Option 2: Include item idx
        new_sum = c_sum + sorted_weights[idx]
        if new_sum < L:
            stack.append((idx + 1, new_sum, mask | (1 << idx)))
            
    # Update best from B&B
    if best_sum_bb > best_sum:
        best_sum = best_sum_bb
        # Reconstruct indices from mask
        best_indices = []
        for i in range(N):
            if best_mask_bb & (1 << i):
                best_indices.append(sorted_indices[i])
                
    # Local Search Refinement
    if time.time() - start_time < time_limit - 5.0:
        # Convert best_indices to a set for fast lookup
        # Note: best_indices are original indices
        selected_set = set(best_indices)
        current_sum = best_sum
        current_deficit = L - current_sum
        
        # Precompute weights for original indices
        # We already have weights list
        
        # Try to improve by swapping
        # To make it fast, we can iterate over a few random restarts or just systematic swaps
        # Systematic swaps O(N^2) is 47^2 ~ 2200 checks. Very fast.
        # We can do multiple passes.
        
        improved = True
        while improved and (time.time() - start_time < time_limit - 2.0):
            improved = False
            # Try removing one and adding one
            for i in list(selected_set):
                w_i = weights[i]
                # Try adding j not in set
                # To be efficient, we can just iterate all j
                for j in range(N):
                    if j in selected_set:
                        continue
                    w_j = weights[j]
                    new_sum = current_sum - w_i + w_j
                    if new_sum < L and new_sum > current_sum:
                        # Accept improvement
                        selected_set.remove(i)
                        selected_set.add(j)
                        current_sum = new_sum
                        improved = True
                        break # Restart loop with new set
                if improved:
                    break
                    
            if improved:
                continue
                
            # Try adding one without removing (if space allows)
            # This is less likely to help if we are close to L, but possible
            for j in range(N):
                if j in selected_set:
                    continue
                w_j = weights[j]
                new_sum = current_sum + w_j
                if new_sum < L and new_sum > current_sum:
                    selected_set.add(j)
                    current_sum = new_sum
                    improved = True
                    break
                    
            if improved:
                continue
            
            # Try removing one without adding (if it brings us closer to L? No, removing decreases sum, increasing deficit)
            # So removing is only useful if we can add something bigger.
            
            # Try removing two and adding one?
            # This is O(N^3) ~ 100k. Still fast.
            # But simple swaps are usually sufficient for this problem structure.
            
            # If no simple swap worked, break?
            # Maybe try a random restart?
            # For now, break if no improvement found in systematic pass.
            break
            
        if current_sum > best_sum:
            best_sum = current_sum
            best_indices = list(selected_set)

    return best_indices
