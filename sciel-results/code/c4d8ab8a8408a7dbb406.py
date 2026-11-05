import time
from fractions import Fraction
import numpy as np

def construct():
    """
    Finds the subset of denominators [2, 3, ..., 48] that minimizes the deficit 
    1 - sum(1/d) subject to sum(1/d) < 1.
    
    Uses float branch-and-bound for deep search, exact rational verification,
    and enhanced local search with multi-swap moves.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Precompute reciprocal values as floats and exact fractions
    reciprocals_float = [1.0 / d for d in denominators]
    reciprocals_frac = [Fraction(1, d) for d in denominators]
    
    # Precompute suffix sums for pruning (float)
    suffix_sums = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sums[i] = suffix_sums[i + 1] + reciprocals_float[i]
    
    # Global best state
    best_sum_float = -1.0
    best_indices = []
    best_sum_frac = None
    
    # Check if a candidate is strictly below 1 using exact arithmetic
    def verify_strictly_below(indices):
        if not indices:
            return Fraction(0, 1)
        s = Fraction(0, 1)
        for idx in indices:
            s += reciprocals_frac[idx]
        return s if s < 1 else None
    
    # Try to improve with exact verification
    def try_improve(indices):
        nonlocal best_sum_float, best_indices, best_sum_frac
        if not indices:
            return
        s = verify_strictly_below(indices)
        if s is not None:
            s_float = float(s)
            if s_float > best_sum_float:
                best_sum_float = s_float
                best_indices = indices[:]
                best_sum_frac = s
    
    # Strategy 1: Greedy initialization - start with small denominators, add while < 1
    greedy_indices = []
    greedy_sum = 0.0
    for i in range(n):
        new_sum = greedy_sum + reciprocals_float[i]
        if new_sum < 1.0:
            greedy_indices.append(i)
            greedy_sum = new_sum
    try_improve(greedy_indices)
    
    # Strategy 2: Enhanced Branch and Bound with tighter pruning
    def dfs(i, cur_sum_float, indices, mask):
        nonlocal best_sum_float, best_indices, best_sum_frac
        
        # Time check
        if time.time() > deadline:
            return
            
        # Pruning 1: If current sum is already >= 1, we can't use this path
        if cur_sum_float >= 1.0:
            return
            
        # Pruning 2: Even if we add all remaining items, can we beat the best?
        if cur_sum_float + suffix_sums[i] <= best_sum_float + 1e-14:
            return
            
        # If we've considered all items, check if this is the best
        if i == n:
            if cur_sum_float > best_sum_float:
                try_improve(indices)
            return
        
        # Option 1: Exclude item i
        dfs(i + 1, cur_sum_float, indices, mask)
        
        # Option 2: Include item i
        new_sum = cur_sum_float + reciprocals_float[i]
        if new_sum < 1.0:
            dfs(i + 1, new_sum, indices + [i], mask)
    
    # Run B&B
    dfs(0, 0.0, [], None)
    
    # Strategy 3: Enhanced Local Search with multi-swap moves
    if best_indices and time.time() < deadline - 10.0:
        current = best_indices[:]
        current_set = set(current)
        current_sum_frac = best_sum_frac
        current_float = float(current_sum_frac)
        
        improved = True
        max_iterations = 100
        iteration = 0
        while improved and iteration < max_iterations:
            improved = False
            iteration += 1
            
            # Try adding any single item not in current
            for i in range(n):
                if i in current_set:
                    continue
                new_sum = current_sum_frac + reciprocals_frac[i]
                if new_sum < 1:
                    if float(new_sum) > current_float:
                        current = current + [i]
                        current_set.add(i)
                        current_sum_frac = new_sum
                        current_float = float(new_sum)
                        improved = True
                        if time.time() > deadline - 3.0:
                            break
            
            if time.time() > deadline - 3.0:
                break
            
            # Try removing one item and adding a different one (1-swap)
            if not improved:
                for remove_idx in current:
                    for add_idx in range(n):
                        if add_idx in current_set:
                            continue
                        new_sum = current_sum_frac - reciprocals_frac[remove_idx] + reciprocals_frac[add_idx]
                        if new_sum < 1 and float(new_sum) > current_float:
                            current.remove(remove_idx)
                            current.append(add_idx)
                            current_set.remove(remove_idx)
                            current_set.add(add_idx)
                            current_sum_frac = new_sum
                            current_float = float(new_sum)
                            improved = True
                            if time.time() > deadline - 3.0:
                                break
                    if time.time() > deadline - 3.0:
                        break
            
            if time.time() > deadline - 3.0:
                break
            
            # Try removing two items and adding two different items (2-swap)
            if not improved and len(current) >= 2:
                current_list = current[:]
                for ri in range(len(current_list)):
                    for rj in range(ri + 1, len(current_list)):
                        remove_i = current_list[ri]
                        remove_j = current_list[rj]
                        base_sum = current_sum_frac - reciprocals_frac[remove_i] - reciprocals_frac[remove_j]
                        for add_i in range(n):
                            if add_i in current_set:
                                continue
                            for add_j in range(n):
                                if add_j in current_set or add_j == add_i:
                                    continue
                                new_sum = base_sum + reciprocals_frac[add_i] + reciprocals_frac[add_j]
                                if new_sum < 1 and float(new_sum) > current_float:
                                    current.remove(remove_i)
                                    current.remove(remove_j)
                                    current.append(add_i)
                                    current.append(add_j)
                                    current_set.discard(remove_i)
                                    current_set.discard(remove_j)
                                    current_set.add(add_i)
                                    current_set.add(add_j)
                                    current_sum_frac = new_sum
                                    current_float = float(new_sum)
                                    improved = True
                                    if time.time() > deadline - 3.0:
                                        break
                            if time.time() > deadline - 3.0:
                                break
                        if time.time() > deadline - 3.0:
                            break
                    if time.time() > deadline - 3.0:
                        break
            
            if time.time() > deadline - 3.0:
                break
            
            # Try adding two items at once
            if not improved:
                for i in range(n):
                    if i in current_set:
                        continue
                    for j in range(i + 1, n):
                        if j in current_set:
                            continue
                        new_sum = current_sum_frac + reciprocals_frac[i] + reciprocals_frac[j]
                        if new_sum < 1 and float(new_sum) > current_float:
                            current = current + [i, j]
                            current_set.add(i)
                            current_set.add(j)
                            current_sum_frac = new_sum
                            current_float = float(new_sum)
                            improved = True
                            if time.time() > deadline - 3.0:
                                break
                    if time.time() > deadline - 3.0:
                        break
        
        try_improve(current)
    
    # Fallback if no solution found
    if not best_indices:
        return [0]
        
    return best_indices

if __name__ == "__main__":
    res = construct()
    pass
