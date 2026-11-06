"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 5040
    target = 4979
    
    # Generate all divisors of n
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors - order doesn't matter for DP correctness
    divisors.sort()
    
    # DP: dp[s] = minimum number of divisors to reach sum s
    # prev[s] = (prev_sum, divisor) to reconstruct
    INF = float('inf')
    dp = [INF] * (target + 1)
    prev = [None] * (target + 1)  # prev[s] = (s - d, d)
    
    dp[0] = 0
    
    for d in divisors:
        if d > target:
            continue
        # 0/1 knapsack: iterate backwards
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                prev[s] = (s - d, d)
    
    if dp[target] == INF:
        # Should not happen, but fallback
        return []
    
    # Reconstruct the solution
    solution = []
    s = target
    while s > 0:
        prev_s, d = prev[s]
        solution.append(d)
        s = prev_s
    
    return solution

# EVOLVE-BLOCK-END
