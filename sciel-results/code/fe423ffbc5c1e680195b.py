"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s limit to be safe (evaluator kills at 90s, but we want margin)
    
    # Compute LCM of 1..22
    L = reduce(lcm, range(1, 23))
    
    # Weights: w[i] = L // (i+1) for i in 0..21
    # Corresponds to x = 1..22
    weights = [L // (i + 1) for i in range(22)]
    
    # Indices sorted by descending weight (ascending x)
    indices_desc = list(range(22))
    indices_desc.sort(key=lambda i: weights[i], reverse=True)
    sorted_weights_desc = [weights[i] for i in indices_desc]
    
    # Indices sorted by ascending weight (descending x)
    indices_asc = list(range(22))
    indices_asc.sort(key=lambda i: weights[i])
    sorted_weights_asc = [weights[i] for i in indices_asc]
    
    best_set = []
    best_size = 0
    
    def update_best(current_mask, current_size):
        nonlocal best_set, best_size
        if current_size > best_size:
            best_size = current_size
            # Convert mask to list of original values
            # mask bit i corresponds to indices_order[i]
            # Wait, we need to map back to original indices 0..21
            # The mask is over the sorted list positions.
            # We need to return the original values 1..22.
            # original_index = sorted_indices[position]
            # value = original_index + 1
            pass # Will implement in search

    def is_valid_addition(current_sums, w):
        """Check if adding weight w to current_sums keeps all sums distinct."""
        new_sums = {s + w for s in current_sums}
        if current_sums & new_sums:
            return False, None
        return True, current_sums | new_sums

    def search(k, start, current_sums, current_mask, current_size, order_indices, order_weights):
        """
        Try to find a subset of size k from order_indices[start:]
        """
        nonlocal best_set, best_size
        
        if time.time() > deadline:
            return False
            
        if current_size == k:
            # Found a valid set of size k
            if k > best_size:
                # Reconstruct set
                s = set()
                for i in range(22):
                    if (current_mask >> i) & 1:
                        s.add(order_indices[i] + 1)
                best_set = list(s)
                best_size = k
            return True
            
        # Pruning: not enough elements left
        if 22 - start < k - current_size:
            return False
            
        # Option 1: Take the next element
        w = order_weights[start]
        valid, new_sums = is_valid_addition(current_sums, w)
        if valid:
            new_mask = current_mask | (1 << start)
            if search(k, start + 1, new_sums, new_mask, current_size + 1, order_indices, order_weights):
                return True
                
        # Option 2: Skip the next element
        if search(k, start + 1, current_sums, current_mask, current_size, order_indices, order_weights):
            return True
            
        return False

    # Try to find the maximum cardinality
    # We try both orders for each k
    for k in range(22, 0, -1):
        if time.time() > deadline:
            break
            
        # Try Descending order
        if search(k, 0, {0}, 0, 0, indices_desc, sorted_weights_desc):
            # Found a set of size k
            break
            
        # Try Ascending order if Descending failed and we have time
        if time.time() < deadline - 5:
            if search(k, 0, {0}, 0, 0, indices_asc, sorted_weights_asc):
                break

    # If no set found, return a default
    if not best_set:
        best_set = [1]
        
    return best_set
# EVOLVE-BLOCK-END
