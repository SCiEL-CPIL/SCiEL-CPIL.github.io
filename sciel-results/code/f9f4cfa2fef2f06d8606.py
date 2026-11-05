"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Generate all distinct positive divisors of 5040
    divisors = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort(reverse=True)
    
    # We want to find a subset of divisors that sums to target, minimizing the number of elements.
    # This is a minimum-cardinality subset sum problem.
    # Since target is 4979 and the largest divisor is 5040, we can't use 5040.
    # The second largest is 2520.
    
    # Dynamic programming approach:
    # dp[s] = minimum number of divisors needed to sum to s
    # We also need to track which divisors were used to reconstruct the solution.
    
    # However, target is 4979, which is manageable for DP.
    # dp[s] = min number of divisors to achieve sum s
    # parent[s] = (prev_sum, divisor_used) for reconstruction
    
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    parent = [None] * (target + 1)  # parent[s] = (prev_sum, d)
    
    for d in divisors:
        if d > target:
            continue
        # Process in reverse to avoid using the same divisor twice (0/1 knapsack style)
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, d)
        
        if time.time() > deadline:
            break
    
    # Check if we found a solution
    if dp[target] == INF:
        # Fallback to greedy
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
                if remainder == 0:
                    break
        return chosen
    
    # Reconstruct the solution
    chosen = []
    s = target
    while s > 0:
        prev_s, d = parent[s]
        chosen.append(d)
        s = prev_s
    
    chosen.sort(reverse=True)
    return chosen
# EVOLVE-BLOCK-END
