import time
from fractions import Fraction
import random

def construct():
    start_time = time.time()
    time_limit = 88.0  # Leave 2s margin

    denominators = list(range(2, 49))
    n = len(denominators)
    values = [Fraction(1, d) for d in denominators]
    float_values = [1.0 / d for d in denominators]

    # Sort indices by value descending for faster "best add" search
    indices_by_val_desc = sorted(range(n), key=lambda x: -float_values[x])

    def get_exact_sum(s):
        return sum(values[i] for i in s)

    def local_search(initial_set, initial_sum_f):
        curr_set = initial_set.copy()
        curr_sum_f = initial_sum_f
        
        # We will iterate until no improvement or time limit
        while (time.time() - start_time) < time_limit:
            improved = False
            
            # 1. Try to add a single item
            limit_add = 1.0 - curr_sum_f
            best_add_j = -1
            best_add_val = 0.0
            # We want the largest v[j] < limit_add
            # Since indices_by_val_desc is sorted, we can break early if v[j] >= limit_add?
            # No, we want the largest v[j] that is STRICTLY less than limit_add.
            # So we iterate from largest to smallest, and pick the first one that fits.
            for j in indices_by_val_desc:
                if j in curr_set:
                    continue
                v_j = float_values[j]
                if v_j < limit_add:
                    best_add_j = j
                    best_add_val = v_j
                    break # Found the largest possible add
            
            if best_add_j != -1:
                curr_set.add(best_add_j)
                curr_sum_f += best_add_val
                improved = True
            
            # 2. Try to swap
            # For each i in curr_set, find best j not in curr_set
            # such that v[j] < 1.0 - curr_sum_f + v[i]
            # and v[j] > v[i] (to improve)
            
            best_swap_delta = 0.0
            best_i = -1
            best_j = -1
            
            # Optimization: We can sort curr_set items by v[i] descending?
            # Doesn't really matter for finding the global best swap in one pass.
            
            for i in list(curr_set):
                v_i = float_values[i]
                limit = 1.0 - curr_sum_f + v_i
                # Find best j not in curr_set with v[j] < limit
                # And we want max (v[j] - v[i]) => max v[j]
                for j in indices_by_val_desc:
                    if j in curr_set:
                        continue
                    v_j = float_values[j]
                    if v_j < limit:
                        delta = v_j - v_i
                        if delta > best_swap_delta:
                            best_swap_delta = delta
                            best_i = i
                            best_j = j
                        break # Since we iterate j in desc order, the first one that fits is the best for this i
            
            if best_swap_delta > 0:
                curr_set.remove(best_i)
                curr_set.add(best_j)
                curr_sum_f += best_swap_delta
                improved = True
                
            if not improved:
                break
                
        # Verify
        exact_sum = get_exact_sum(curr_set)
        if exact_sum < 1:
            return curr_set, curr_sum_f, exact_sum
        else:
            # Invalid due to float precision
            # Return empty or best found? 
            # For local search, we just return None to indicate failure
            return None, 0, 1

    # Initial Greedy
    best_set, best_sum_f = set(), 0.0
    for i in indices_by_val_desc:
        if best_sum_f + float_values[i] < 1.0:
            best_set.add(i)
            best_sum_f += float_values[i]
            
    # Refine with local search
    res = local_search(best_set, best_sum_f)
    if res:
        best_set, best_sum_f, best_exact = res
    else:
        best_exact = get_exact_sum(best_set)
        
    best_deficit = 1 - best_exact
    
    # Random Restart / Perturbation
    while (time.time() - start_time) < time_limit:
        # Perturb best_set
        perturbed_set = best_set.copy()
        # Remove 1 or 2 random items
        num_remove = random.randint(1, 2)
        if len(perturbed_set) >= num_remove:
            to_remove = random.sample(list(perturbed_set), num_remove)
            for i in to_remove:
                perturbed_set.remove(i)
        
        # Recalculate float sum
        p_sum_f = sum(float_values[i] for i in perturbed_set)
        
        # Local search
        res = local_search(perturbed_set, p_sum_f)
        if res:
            p_set, p_sum_f, p_exact = res
            p_deficit = 1 - p_exact
            if p_deficit < best_deficit:
                best_deficit = p_deficit
                best_set = p_set
                best_sum_f = p_sum_f
                best_exact = p_exact
        else:
            # If local search failed (invalid), try a fully random start?
            pass

    return sorted(list(best_set))
