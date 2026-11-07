import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    max_val = 256
    max_size = 60
    
    def _try_greedy(order, deadline):
        """
        Attempts to build a valid set by adding elements in the given order.
        Uses bitset DP for fast validity checks.
        Returns the constructed list of integers.
        """
        # masks[k] is a bitset where bit s is 1 if sum s is reachable with k elements
        # Max sum approx 60 * 256 = 15360. Python ints handle this easily.
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
            # Collision with masks[i+1] is allowed (same cardinality).
            
            valid = True
            # i is the source cardinality index in old masks
            # Max source index is len(current_set)
            for i in range(len(current_set) + 1):
                if masks[i] == 0:
                    continue
                
                new_sums = masks[i] << x
                target_k = i + 1
                
                # Check collision with all other cardinalities j != target_k
                # j ranges from 0 to len(current_set)
                for j in range(len(current_set) + 1):
                    if j == target_k:
                        continue
                    if masks[j] != 0 and (new_sums & masks[j]):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Update masks
                # Update in reverse order to avoid using x multiple times
                # masks[k] |= masks[k-1] << x
                for k in range(len(current_set) + 1, 0, -1):
                    masks[k] |= (masks[k-1] << x)
                current_set.append(x)
        
        return current_set

    best_set = []
    
    # Strategy 1: Descending order (spread sums)
    order_desc = list(range(max_val, 0, -1))
    res = _try_greedy(order_desc, deadline)
    if len(res) > len(best_set):
        best_set = res
        
    # Strategy 2: Ascending order
    order_asc = list(range(1, max_val + 1))
    if time.time() < deadline:
        res = _try_greedy(order_asc, deadline)
        if len(res) > len(best_set):
            best_set = res
            
    # Strategy 3: Random orders
    # If we have time, try several random permutations
    # Since the bitset check is very fast, we can try many.
    attempts = 0
    while time.time() < deadline - 5:
        random_order = list(range(1, max_val + 1))
        random.shuffle(random_order)
        res = _try_greedy(random_order, deadline)
        if len(res) > len(best_set):
            best_set = res
            # If we reached max size, we can stop
            if len(best_set) >= max_size:
                break
        attempts += 1
        # Limit attempts to avoid infinite loop if something is wrong, though time check handles it
        if attempts > 10000:
            break

    # Final validation and return
    if not best_set:
        # Fallback to superincreasing
        best_set = [1, 2, 4, 8, 16, 32, 64, 128]
        
    # Convert to list of ints
    return list(best_set)

# EVOLVE-BLOCK-END
