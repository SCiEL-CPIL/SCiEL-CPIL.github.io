import math
import time

def construct():
    n = 5040
    target = 4979
    deadline = time.time() + 50.0  # 50s limit to be safe

    # Generate divisors of 5040
    divisors = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Filter and sort
    divisors = [d for d in divisors if d <= target]
    divisors.sort()
    N = len(divisors)

    # DP for min items
    INF = 10**9
    dp = [INF] * (target + 1)
    dp[0] = 0
    parent = [0] * (target + 1)
    
    # Standard 0/1 knapsack for min items
    for d in divisors:
        if time.time() > deadline:
            break
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = d

    if dp[target] >= INF:
        return []

    # Reconstruct
    chosen = []
    s = target
    while s > 0:
        d_val = parent[s]
        if d_val == 0:
            break
        chosen.append(d_val)
        s -= d_val
        
    # Verification
    if sum(chosen) != target or len(set(chosen)) != len(chosen):
        return []
        
    return chosen
