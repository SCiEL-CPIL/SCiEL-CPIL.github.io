"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            divisors.append(d)
    
    # Sort in descending order for potential early termination benefits
    divisors.sort(reverse=True)
    m = len(divisors)  # 60 divisors
    
    # DP: dp[s] = minimum number of divisors needed to achieve sum s
    # Initialize with infinity, dp[0] = 0
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # Also track parent for reconstruction
    # parent[s] = (previous_sum, divisor_used) or None
    parent = [None] * (target + 1)
    
    # Process each divisor (0/1 knapsack)
    for d in divisors:
        # Process in reverse to ensure each divisor used at most once
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, d)
    
    # Check if target is reachable
    if dp[target] == INF:
        # Fallback: greedy (should not happen for this instance)
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
        assert remainder == 0, f"Greedy failed, remainder={remainder}"
        return chosen
    
    # Reconstruct the solution
    chosen = []
    s = target
    while s > 0:
        prev_s, d = parent[s]
        chosen.append(d)
        s = prev_s
    
    chosen.reverse()
    
    # Verify
    assert len(set(chosen)) == len(chosen), "Not distinct"
    assert all(n % d == 0 for d in chosen), "Not all divisors of 5040"
    assert sum(chosen) == target, f"Sum mismatch: {sum(chosen)} != {target}"
    assert len(chosen) <= 256, f"Too many divisors: {len(chosen)}"
    
    return chosen
# EVOLVE-BLOCK-END
