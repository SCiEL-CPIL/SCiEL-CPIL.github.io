import time
import numpy as np

_N = 5040
_TARGET = 4979
_DEADLINE = 85.0  # Safety margin below 90s kill

def _divisors_of(n):
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
    return sorted(divs)

def construct():
    start = time.time()
    
    # Get divisors of 5040 that are <= target
    divisors = [d for d in _divisors_of(_N) if d <= _TARGET]
    
    # DP for minimum cardinality subset sum
    # dp[s] = minimum number of divisors needed to sum to s
    # parent[s] = the divisor that was added to reach s optimally
    INF = 10**9
    dp = [INF] * (_TARGET + 1)
    parent = [-1] * (_TARGET + 1)
    dp[0] = 0
    
    # Use numpy for vectorized inner loop where possible for speedup
    # However, the standard DP loop is already quite fast for this size.
    # Let's stick to the optimized Python loop first, as numpy overhead might not be worth it for 5000x250 ops.
    # Actually, 5000 * 250 = 1,250,000 operations. Python can do this in < 0.1s.
    # The bottleneck is not the DP itself.
    
    # Wait, the task says "Minimize the number of selected divisors".
    # The DP approach finds the minimum number of divisors to sum to target.
    # This is exactly what we want.
    
    for d in divisors:
        if time.time() > _DEADLINE - 5:
            break
        for s in range(_TARGET, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = d
    
    # Check if we found a solution
    if dp[_TARGET] < INF:
        result = []
        s = _TARGET
        while s > 0:
            d = parent[s]
            if d == -1:
                break
            result.append(d)
            s -= d
        
        # Verify the solution
        if len(result) > 0 and sum(result) == _TARGET and len(set(result)) == len(result):
            return result
    
    # Fallback: greedy descending (guaranteed to work for practical numbers)
    remainder = _TARGET
    chosen = []
    for d in sorted(divisors, reverse=True):
        if d <= remainder:
            chosen.append(d)
            remainder -= d
            if remainder == 0:
                break
    
    if sum(chosen) == _TARGET and len(set(chosen)) == len(chosen):
        return chosen
    
    # Ultimate fallback: should never reach here
    return []
