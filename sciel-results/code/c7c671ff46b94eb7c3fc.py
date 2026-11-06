import time

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort()
    
    # DP to find minimum number of distinct divisors summing to target
    # dp[s] = min number of divisors to sum to s
    # parent[s] = previous sum state
    # used_divisor[s] = the divisor added to reach state s
    
    INF = float('inf')
    dp = [INF] * (target + 1)
    parent = [-1] * (target + 1)
    used_divisor = [-1] * (target + 1)
    
    dp[0] = 0
    
    # Iterate through each divisor
    for d in divisors:
        # Iterate sums backwards to ensure each divisor is used at most once (0/1 Knapsack)
        # This guarantees distinct divisors
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = s - d
                used_divisor[s] = d
                
    if dp[target] == INF:
        # Fallback: Should not happen for practical number 5040
        # But just in case, return a valid subset if found via simple check, else empty
        # We know a solution exists.
        return []
    
    # Reconstruct the solution
    result = []
    s = target
    while s > 0:
        d = used_divisor[s]
        if d == -1:
            # Reconstruction error
            break
        result.append(d)
        s = parent[s]
        
    # Verification steps
    if sum(result) != target:
        return []
        
    if len(result) != len(set(result)):
        return []
        
    for d in result:
        if 5040 % d != 0:
            return []
            
    return result
