import time

def construct():
    deadline = time.time() + 55.0
    
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    divisor_set = set(divisors)
    
    # Check for count 1
    if target in divisor_set:
        return [target]
        
    # Check for count 2
    for d1 in divisors:
        d2 = target - d1
        if d2 in divisor_set and d2 != d1:
            return [d1, d2]
            
    # Check for count 3
    # To avoid duplicates and ensure distinctness, we can just iterate all pairs
    # Since K is small, O(K^2) is fine.
    for i, d1 in enumerate(divisors):
        for d2 in divisors[i+1:]:
            d3 = target - d1 - d2
            if d3 in divisor_set and d3 != d1 and d3 != d2:
                return [d1, d2, d3]
                
    # If not found in 3, use DP for minimum count
    INF = float('inf')
    dp = [INF] * (target + 1)
    parent = [None] * (target + 1)
    dp[0] = 0
    
    for d in divisors:
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (d, s - d)
                
        if time.time() > deadline:
            break
            
    if dp[target] == INF:
        # Fallback: Greedy (should not happen if solution exists)
        remainder = target
        chosen = []
        for d in sorted(divisors, reverse=True):
            if d <= remainder:
                chosen.append(d)
                remainder -= d
        if remainder == 0:
            return chosen
        # If even greedy fails, return empty or raise
        return []
        
    # Reconstruct
    chosen = []
    s = target
    while s > 0:
        d, prev_s = parent[s]
        chosen.append(d)
        s = prev_s
        
    return chosen
