import time
from math import lcm
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # LCM of 1..22
    L = reduce(lcm, range(1, 23))
    
    # Reciprocals as integers
    # v[i] corresponds to 1/(i+1)
    v = [L // (i + 1) for i in range(22)]
    
    # Sort by value descending (largest first) to try safer elements first
    # Track original indices
    order = sorted(range(22), key=lambda i: v[i], reverse=True)
    
    # We want to find the largest k such that there exists a subset of size k
    # that is dissociated.
    
    # Let's iterate k from 22 down to 1
    for k in range(22, 0, -1):
        # Try to find a subset of size k
        found = False
        result_indices = []
        
        def dfs(pos, current_sums, current_count, chosen_indices):
            nonlocal found, result_indices
            if found:
                return
            if time.time() > deadline:
                return
            
            # Pruning: not enough elements left
            if current_count + (22 - pos) < k:
                return
            
            if current_count == k:
                found = True
                result_indices = list(chosen_indices)
                return
            
            idx = order[pos]
            val = v[idx]
            
            # Try INCLUDE first (bias toward larger sets)
            # Check if including val keeps sums distinct
            new_sums = set()
            for s in current_sums:
                new_sums.add(s + val)
            
            if current_sums.isdisjoint(new_sums):
                next_sums = current_sums | new_sums
                chosen_indices.append(idx)
                dfs(pos + 1, next_sums, current_count + 1, chosen_indices)
                chosen_indices.pop()
                if found:
                    return
            
            # Try SKIP
            dfs(pos + 1, current_sums, current_count, chosen_indices)
            if found:
                return
        
        # Initial call
        dfs(0, {0}, 0, [])
        
        if found:
            # result_indices contains indices 0..21
            # Convert to 1..22
            return [i + 1 for i in result_indices]
        
    # Fallback
    return [1]
