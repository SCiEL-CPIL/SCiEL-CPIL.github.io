"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Get all divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort(reverse=True)
    
    INF = float('inf')
    max_sum = target
    
    # dp[s] = min number of divisors to reach sum s
    dp = [INF] * (max_sum + 1)
    dp[0] = 0
    
    # prev_sum[s] = the sum before adding the last divisor
    # prev_div_idx[s] = index of the last divisor added
    prev_sum = [-1] * (max_sum + 1)
    prev_div_idx = [-1] * (max_sum + 1)
    
    for idx, d in enumerate(divisors):
        if time.time() > deadline:
            break
        # Process in reverse to ensure each divisor is used at most once
        for s in range(max_sum, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                prev_sum[s] = s - d
                prev_div_idx[s] = idx
    
    if dp[target] == INF:
        # Fallback: greedy (should not happen for practical numbers within range)
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
        idx = prev_div_idx[s]
        if idx == -1:
            break # Safety check
        chosen.append(divisors[idx])
        s = prev_sum[s]
    
    return chosen

# EVOLVE-BLOCK-END
