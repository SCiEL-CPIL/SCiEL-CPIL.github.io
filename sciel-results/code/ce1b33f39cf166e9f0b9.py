import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin of 5s
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    sorted_denoms = sorted(denominators)
    sorted_recips = [Fraction(1, d) for d in sorted_denoms]
    
    suffix_sum = [Fraction(0)] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + sorted_recips[i]
    
    best_sum = Fraction(0)
    best_indices = []
    
    node_count = 0
    CHECK_INTERVAL = 1000
    
    def dfs(idx, current_sum, chosen_denoms):
        nonlocal best_sum, best_indices, node_count
        
        node_count += 1
        if node_count % CHECK_INTERVAL == 0:
            if time.time() > deadline:
                return
            
        if current_sum >= 1:
            return
            
        if current_sum + suffix_sum[idx] <= best_sum:
            return
            
        if idx == n:
            if current_sum > best_sum:
                best_sum = current_sum
                best_indices = list(chosen_denoms)
            return
        
        dfs(idx + 1, current_sum, chosen_denoms)
        
        new_sum = current_sum + sorted_recips[idx]
        if new_sum < 1:
            chosen_denoms.append(sorted_denoms[idx])
            dfs(idx + 1, new_sum, chosen_denoms)
            chosen_denoms.pop()
            
    dfs(0, Fraction(0), [])
    
    result_indices = [d - 2 for d in best_indices]
    
    return result_indices
