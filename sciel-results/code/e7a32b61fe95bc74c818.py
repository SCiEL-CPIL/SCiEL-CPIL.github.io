"""Find a dissociated reciprocal set from [1,22] maximizing cardinality."""
import time
import random
from math import lcm

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s safety margin before 90s kill
    
    n = 22
    # Calculate LCM of 1..22 to convert to integer sums
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
    
    # Map each x to its integer reciprocal value
    vals = {x: L // x for x in range(1, n + 1)}
    
    best_result = [1]  # fallback
    best_k = 1
    
    def _get_sums(v_list):
        """Calculate the set of all subset sums for a list of values."""
        if not v_list:
            return {0}
        sums = {0}
        for v in v_list:
            shifted = {s + v for s in sums}
            if shifted & sums:
                return None # Collision detected early
            sums |= shifted
        return sums

    def _check_dissociated(v_list):
        """Check if all subset sums of v_list are distinct."""
        return _get_sums(v_list) is not None

    def _greedy_build(order):
        """Greedy: add elements in given order if they keep the set dissociated."""
        result = []
        result_sums = {0}
        for idx in order:
            v = vals[idx]
            shifted = {s + v for s in result_sums}
            if not (shifted & result_sums):
                result.append(idx)
                result_sums |= shifted
        return result, result_sums

    # Try multiple orderings to get good starting points
    orderings = []
    orderings.append(sorted(range(1, n + 1), key=lambda x: -vals[x]))
    orderings.append(sorted(range(1, n + 1), key=lambda x: vals[x]))
    orderings.append(list(range(1, n + 1)))
    orderings.append(list(range(n, 0, -1)))
    
    rng = random.Random(42)
    for _ in range(10):
        random_order = list(range(1, n + 1))
        rng.shuffle(random_order)
        orderings.append(random_order)
    
    # Initial Greedy Pass
    for order in orderings:
        if time.time() > deadline:
            break
        greedy_set, _ = _greedy_build(order)
        if len(greedy_set) > best_k:
            best_k = len(greedy_set)
            best_result = list(greedy_set)
    
    # Local Search: Try to improve by "Add 1" or "Remove 1, Add 2"
    current_set = list(best_result)
    current_k = len(current_set)
    
    improved = True
    max_iterations = 50
    iter_count = 0
    
    while improved and time.time() < deadline - 5.0 and iter_count < max_iterations:
        improved = False
        iter_count += 1
        
        in_set = set(current_set)
        not_in_set = [x for x in range(1, n + 1) if x not in in_set]
        
        # 1. Try to add a single element
        for x in not_in_set:
            if time.time() > deadline - 5.0:
                break
            test_set = current_set + [x]
            if _check_dissociated([vals[y] for y in test_set]):
                current_set = test_set
                current_k += 1
                in_set.add(x)
                not_in_set.remove(x)
                improved = True
                break
        
        if improved:
            if current_k > best_k:
                best_k = current_k
                best_result = list(current_set)
            continue
        
        # 2. Try to Remove 1, Add 2 (Net +1)
        # To limit search space, we can shuffle not_in_set and current_set
        random.shuffle(not_in_set)
        random.shuffle(current_set) # This changes the list order, but set membership is what matters for the check
        
        found_move = False
        for y in current_set:
            if time.time() > deadline - 5.0:
                break
            if found_move:
                break
            
            # Base set without y
            base_set = [z for z in current_set if z != y]
            base_vals = [vals[z] for z in base_set]
            base_sums = _get_sums(base_vals)
            
            if base_sums is None:
                continue # Should not happen if current_set is valid
            
            # Try adding pairs from not_in_set
            for i in range(len(not_in_set)):
                if time.time() > deadline - 5.0:
                    break
                x1 = not_in_set[i]
                v1 = vals[x1]
                
                # Add x1
                sums_with_x1 = base_sums | {s + v1 for s in base_sums}
                if len(sums_with_x1) < 2 * len(base_sums):
                    continue # Collision with x1
                
                for j in range(i + 1, len(not_in_set)):
                    if time.time() > deadline - 5.0:
                        break
                    x2 = not_in_set[j]
                    v2 = vals[x2]
                    
                    # Add x2
                    # Check if sums_with_x1 and sums_with_x1 + v2 are disjoint
                    shifted_x2 = {s + v2 for s in sums_with_x1}
                    if not (shifted_x2 & sums_with_x1):
                        # Valid move
                        new_set = base_set + [x1, x2]
                        current_set = new_set
                        current_k += 1
                        in_set = set(current_set)
                        not_in_set = [x for x in range(1, n + 1) if x not in in_set]
                        improved = True
                        found_move = True
                        if current_k > best_k:
                            best_k = current_k
                            best_result = list(current_set)
                        break

    return best_result
# EVOLVE-BLOCK-END
