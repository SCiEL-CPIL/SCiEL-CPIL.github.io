import time
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 90s limit

    # Denominators 2 to 48 inclusive
    denominators = list(range(2, 49))
    n = len(denominators)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    # Compute LCM of 2..48
    L = reduce(lcm, range(2, 49))
    
    # Precompute integer weights
    # index i corresponds to denominator i+2
    # weight[i] = L // (i+2)
    weights = [L // d for d in denominators]
    
    # Sort indices by weight descending to improve B&B pruning
    # weights are already descending because denominators are ascending
    # w[0] = L/2, w[1] = L/3, etc.
    
    # 1. Greedy Initial Solution
    current_sum = 0
    selected = [False] * n
    for i in range(n):
        if current_sum + weights[i] < L:
            current_sum += weights[i]
            selected[i] = True
    
    best_sum = current_sum
    best_indices = [i for i in range(n) if selected[i]]
    
    # 2. Local Search to improve initial bound
    # Try swaps: remove i, add j
    # Try add: add j (if valid)
    # Try remove: remove i (if it allows adding larger items later? No, we maximize sum)
    
    def local_search():
        nonlocal best_sum, best_indices
        improved = True
        while improved:
            improved = False
            current_sel = set(best_indices)
            current_sum_val = best_sum
            
            # Try adding any unselected item
            for j in range(n):
                if j not in current_sel:
                    if current_sum_val + weights[j] < L:
                        new_sum = current_sum_val + weights[j]
                        if new_sum > best_sum:
                            best_sum = new_sum
                            best_indices = list(current_sel | {j})
                            improved = True
                            break # Restart local search with new best
            
            if improved:
                continue
                
            # Try swapping: remove i, add j
            # Only if weights[j] > weights[i] to ensure potential improvement
            for i in best_indices:
                for j in range(n):
                    if j not in current_sel and weights[j] > weights[i]:
                        new_sum = current_sum_val - weights[i] + weights[j]
                        if new_sum < L and new_sum > best_sum:
                            best_sum = new_sum
                            new_set = set(best_indices)
                            new_set.remove(i)
                            new_set.add(j)
                            best_indices = list(new_set)
                            improved = True
                            break # Restart local search
                if improved:
                    break
                    
    local_search()
    
    # 3. Branch and Bound
    # Sort items by weight descending (already done)
    # Suffix sums for pruning
    suffix_sum = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + weights[i]
    
    best_sum_weight = best_sum
    best_indices_final = list(best_indices)
    
    def dfs(idx, current_sum, current_indices):
        nonlocal best_sum_weight, best_indices_final
        
        if time.time() - start_time > timeout:
            return
            
        # Pruning: even if we take all remaining, we can't beat best
        if current_sum + suffix_sum[idx] <= best_sum_weight:
            return
            
        if idx == n:
            if current_sum > best_sum_weight:
                best_sum_weight = current_sum
                best_indices_final = list(current_indices)
            return
            
        # Option 1: Include item idx
        if current_sum + weights[idx] < L:
            current_indices.append(idx)
            dfs(idx + 1, current_sum + weights[idx], current_indices)
            current_indices.pop()
            
        if time.time() - start_time > timeout:
            return
            
        # Option 2: Exclude item idx
        dfs(idx + 1, current_sum, current_indices)
        
    dfs(0, 0, [])
    
    if not best_indices_final:
        return [0]
        
    return best_indices_final
