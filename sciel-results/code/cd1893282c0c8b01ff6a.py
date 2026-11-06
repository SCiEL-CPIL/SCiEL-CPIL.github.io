import time

def construct():
    n = 5040
    target = 4979
    start_time = time.time()
    # 90s limit, use 85s as deadline to be safe
    deadline = start_time + 85.0

    # Generate divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    divisors.sort(reverse=True)
    
    # Precompute suffix sums for pruning
    suffix_sum = [0] * (len(divisors) + 1)
    for i in range(len(divisors) - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + divisors[i]
        
    best_count = float('inf')
    best_set = []
    timed_out = False
    
    def dfs(index, current_sum, current_set):
        nonlocal best_count, best_set, timed_out
        
        if timed_out:
            return
            
        if time.time() > deadline:
            timed_out = True
            return
            
        # Pruning: Current set size is already >= best known, no need to explore deeper
        # Note: We are looking for MINIMUM count.
        # If we have k items and best is k, we can't improve.
        # If we have k items and best is k+1, we might improve to k.
        # So if len(current_set) >= best_count, we can stop.
        if len(current_set) >= best_count:
            return
            
        if current_sum == target:
            if len(current_set) < best_count:
                best_count = len(current_set)
                best_set = current_set.copy()
            return
            
        if current_sum > target:
            return
            
        if index >= len(divisors):
            return
            
        # Pruning: Even if we take all remaining, we can't reach target
        if suffix_sum[index] < (target - current_sum):
            return
            
        # Branch 1: Include divisors[index]
        current_set.append(divisors[index])
        dfs(index + 1, current_sum + divisors[index], current_set)
        current_set.pop()
        
        # Branch 2: Exclude divisors[index]
        dfs(index + 1, current_sum, current_set)

    dfs(0, 0, [])
    
    if best_count == float('inf'):
        # If no solution found (should not happen), return empty or raise
        # The task implies a solution exists.
        # If we timed out, we might have a partial best_set?
        # No, best_set is only updated on exact match.
        # If timed out before finding any, we fail.
        # But 5040 is small.
        pass
        
    return best_set
