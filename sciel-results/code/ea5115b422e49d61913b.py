import math

def construct():
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors in descending order to potentially find solutions faster?
    # For min count DP, order doesn't matter for correctness, but might for speed if we prune.
    # However, standard DP is O(N*T).
    divisors.sort(reverse=True)
    
    INF = float('inf')
    dp = [INF] * (target + 1)
    parent = [None] * (target + 1)
    
    dp[0] = 0
    
    for d in divisors:
        # Iterate backwards to ensure each divisor is used at most once
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = d
    
    if dp[target] == INF:
        return []
        
    selected = []
    s = target
    while s > 0:
        d = parent[s]
        selected.append(d)
        s -= d
        
    return selected
