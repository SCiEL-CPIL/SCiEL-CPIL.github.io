import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    max_val = 256
    max_size = 60
    
    # Helper to check if a set is valid (for final verification)
    def _check_validity(s):
        n = len(s)
        if n == 0:
            return True
        # sums[k] = set of all sums using exactly k elements from s
        sums = [set() for _ in range(n + 1)]
        sums[0].add(0)
        
        for x in s:
            # Update from high to low to avoid using x multiple times
            for k in range(n, 0, -1):
                if sums[k-1]:
                    for prev_sum in sums[k-1]:
                        sums[k].add(prev_sum + x)
        
        # Check that all sums[k] are pairwise disjoint
        seen = set()
        for k in range(n + 1):
            for val in sums[k]:
                if val in seen:
                    return False
                seen.add(val)
        return True

    # Bitset-based greedy attempt
    def _try_greedy(order, deadline):
        """
        Attempts to build a valid set by adding elements in the given order.
        Uses bitset DP for fast validity checks.
        """
        masks = [0] * (max_size + 1)
        masks[0] = 1  # Sum 0 with 0 elements
        
        current_set = []
        
        for x in order:
            if len(current_set) >= max_size:
                break
            if time.time() > deadline:
                break
            
            # Check if adding x is valid
            # For each source cardinality i, new sums are masks[i] << x.
            # These new sums will be part of cardinality i+1.
            # They must not collide with masks[j] for any j != i+1.
            
            valid = True
            n_curr = len(current_set)
            
            # Pre-compute union of all masks for fast forbidden check?
            # all_sums = OR of masks[0..n_curr]
            all_sums = 0
            for j in range(n_curr + 1):
                all_sums |= masks[j]
            
            for i in range(n_curr + 1):
                if masks[i] == 0:
                    continue
                
                new_sums = masks[i] << x
                target_k = i + 1
                
                # Forbidden bits: all_sums excluding bits already in masks[target_k]
                # Note: masks[target_k] might be 0 if target_k > n_curr
                forbidden = all_sums
                if target_k <= n_curr:
                    forbidden ^= masks[target_k]
                
                if new_sums & forbidden:
                    valid = False
                    break
            
            if valid:
                # Update masks
                for k in range(n_curr + 1, 0, -1):
                    masks[k] |= (masks[k-1] << x)
                current_set.append(x)
        
        return current_set

    best_set = []
    
    # Strategy 1: Descending order
    order_desc = list(range(max_val, 0, -1))
    res = _try_greedy(order_desc, deadline)
    if len(res) > len(best_set):
        best_set = res[:]
        
    if time.time() > deadline - 2:
        return [int(x) for x in best_set]

    # Strategy 2: Ascending order
    order_asc = list(range(1, max_val + 1))
    res = _try_greedy(order_asc, deadline)
    if len(res) > len(best_set):
        best_set = res[:]
        
    if time.time() > deadline - 2:
        return [int(x) for x in best_set]

    # Strategy 3: Random restarts
    for _ in range(20):
        if time.time() > deadline - 1:
            break
        random.shuffle(order_desc)
        res = _try_greedy(order_desc, deadline)
        if len(res) > len(best_set):
            best_set = res[:]
            
    # Final verification
    if not _check_validity(best_set):
        # Fallback to powers of 2
        best_set = [2**i for i in range(9) if 2**i <= 256]
        
    return [int(x) for x in best_set]
