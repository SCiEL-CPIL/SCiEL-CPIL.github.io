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
    
    # Filter to only divisors <= target
    divisors = [d for d in divisors if d <= target]
    
    # Sort divisors
    divisors.sort()
    
    # DP: dp[s] = minimum number of divisors to reach sum s
    # prev[s] = (prev_sum, divisor) to reconstruct
    INF = float('inf')
    dp = [INF] * (target + 1)
    prev = [None] * (target + 1)
    
    dp[0] = 0
    
    # Deadline guard
    start_time = time.time()
    deadline = 60.0 # 60 seconds limit
    
    for d in divisors:
        if time.time() > deadline - 5.0:
            break
        if d > target:
            continue
        # 0/1 knapsack: iterate backwards
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                prev[s] = (s - d, d)
    
    if dp[target] == INF:
        return []
    
    # Reconstruct the solution
    solution = []
    s = target
    while s > 0:
        if prev[s] is None:
            return []
        prev_s, d = prev[s]
        solution.append(d)
        s = prev_s
    
    return solution

# EVOLVE-BLOCK-END
