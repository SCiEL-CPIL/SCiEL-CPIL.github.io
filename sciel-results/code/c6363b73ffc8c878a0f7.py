"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    deadline = time.time() + 85.0  # safety margin before 90s limit
    
    # Generate all divisors of 5040
    n = 5040
    target = 4979
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors in descending order for potential greedy improvements
    divisors.sort(reverse=True)
    
    # DP: dp[s] = minimum number of distinct divisors to sum to s
    # Use a large array
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # 0/1 knapsack: for each divisor, update dp in reverse
    for d in divisors:
        if d > target:
            continue
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
    
    if dp[target] == INF:
        # Fallback: greedy (should not happen)
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
        assert remainder == 0
        return chosen
    
    # Reconstruct the solution
    # We need to track which divisors were used
    # Redo DP with parent tracking
    dp = [INF] * (target + 1)
    dp[0] = 0
    parent = [None] * (target + 1)  # parent[s] = (prev_sum, divisor_used)
    
    for d in divisors:
        if d > target:
            continue
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, d)
    
    # Reconstruct
    chosen = []
    s = target
    while s > 0:
        if parent[s] is None:
            break
        prev_s, d = parent[s]
        chosen.append(d)
        s = prev_s
    
    # Verify
    assert len(set(chosen)) == len(chosen), "Divisors must be distinct"
    assert sum(chosen) == target, f"Sum mismatch: {sum(chosen)} != {target}"
    for d in chosen:
        assert n % d == 0, f"{d} is not a divisor of {n}"
    
    return chosen

# EVOLVE-BLOCK-END
