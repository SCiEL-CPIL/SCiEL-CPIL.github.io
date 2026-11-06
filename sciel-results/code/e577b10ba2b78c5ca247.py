"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 55.0  # safety margin before 60s limit
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            divisors.append(d)
    
    # DP for minimum count subset sum
    # dp[s] = minimum number of divisors to achieve sum s
    INF = float('inf')
    dp = [INF] * (target + 1)
    parent = [None] * (target + 1)  # parent[s] = (divisor_used, prev_sum)
    dp[0] = 0
    
    for d in divisors:
        # Process in reverse to ensure 0/1 constraint
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (d, s - d)
        
        # Check deadline
        if time.time() > deadline:
            break
    
    if dp[target] == INF:
        # Fallback to greedy
        remainder = target
        chosen = []
        for d in range(n, 0, -1):
            if n % d == 0 and d <= remainder:
                chosen.append(d)
                remainder -= d
        assert remainder == 0
        return chosen
    
    # Reconstruct solution
    chosen = []
    s = target
    while s > 0:
        d, prev_s = parent[s]
        chosen.append(d)
        s = prev_s
    
    # Verify
    assert len(chosen) <= 256
    assert len(set(chosen)) == len(chosen), "Divisors must be distinct"
    assert sum(chosen) == target
    for x in chosen:
        assert n % x == 0, f"{x} is not a divisor of {n}"
    
    return chosen
# EVOLVE-BLOCK-END
