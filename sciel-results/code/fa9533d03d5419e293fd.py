import time
import math

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin for 60s limit
    
    N = 22
    # Calculate LCM of 1..22
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    M = 1
    for i in range(1, N + 1):
        M = lcm(M, i)
        
    # Weights
    weights = [M // i for i in range(1, N + 1)]
    
    # Greedy fallback
    def greedy_construct():
        selected = []
        sums = {0}
        diffs = {0}
        # Try adding elements in order of increasing weight? Or decreasing?
        # Let's try increasing index (1, 2, 3...)
        for i in range(N):
            w = weights[i]
            if w not in diffs:
                # Check if we can add
                # New sums
                new_sums = set(sums)
                for s in sums:
                    new_sums.add(s + w)
                # New diffs
                new_diffs = set(diffs)
                for s in sums:
                    for t in sums:
                        new_diffs.add(abs(s + w - t))
                selected.append(i)
                sums = new_sums
                diffs = new_diffs
        return [idx + 1 for idx in selected]

    best_greedy = greedy_construct()
    best_size = len(best_greedy)
    
    # Backtracking to find max size
    def backtrack(idx, current_sums, current_diffs, selected, target_size):
        if time.time() - start_time > timeout:
            return None
            
        if len(selected) == target_size:
            return selected[:]
            
        if idx == N:
            return None
            
        # Option 1: Include weights[idx]
        w = weights[idx]
        if w not in current_diffs:
            new_sums = set(current_sums)
            for s in current_sums:
                new_sums.add(s + w)
                
            new_diffs = set(current_diffs)
            for s in current_sums:
                for t in current_sums:
                    new_diffs.add(abs(s + w - t))
            
            result = backtrack(idx + 1, new_sums, new_diffs, selected + [idx], target_size)
            if result is not None:
                return result
            
        # Option 2: Exclude weights[idx]
        result = backtrack(idx + 1, current_sums, current_diffs, selected, target_size)
        if result is not None:
            return result
            
        return None

    # Try to find a subset of size k, starting from N down to best_size + 1
    # If we find one, it's better than greedy.
    # If we don't find one, we return greedy.
    
    for k in range(N, best_size, -1):
        # If k is too large, the search might be too slow.
        # We can add a heuristic: if k > 15, maybe skip? 
        # But let's try. The time guard will stop it.
        
        res = backtrack(0, {0}, {0}, [], k)
        
        if res is not None:
            subset = [idx + 1 for idx in res]
            return subset
            
        # If we get here, no solution of size k found (or timeout).
        # If timeout, we should break and return best_greedy.
        if time.time() - start_time > timeout - 5:
            break

    return best_greedy

