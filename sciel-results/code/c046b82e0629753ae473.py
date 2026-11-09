"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    t0 = time.time()
    DEADLINE = 50.0  # safety margin under 60s limit
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for i in range(1, int(n ** 0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    divisors.sort(reverse=True)  # larger first for better pruning
    m = len(divisors)
    
    # DP: dp[s] = (min_count, parent_info) to achieve sum s
    # parent_info tracks which divisor was added to reach s
    # We'll use: dp[s] = (count, prev_sum, divisor_used)
    
    INF = float('inf')
    dp_count = [INF] * (target + 1)
    dp_parent = [(-1, -1)] * (target + 1)  # (prev_sum, divisor)
    
    dp_count[0] = 0
    
    for d in divisors:
        if d > target:
            continue
        # Update in reverse to avoid using same divisor twice
        for s in range(target, d - 1, -1):
            if dp_count[s - d] + 1 < dp_count[s]:
                dp_count[s] = dp_count[s - d] + 1
                dp_parent[s] = (s - d, d)
    
    if dp_count[target] == INF:
        # Fallback: greedy
        rem = target
        chosen = []
        for d in divisors:
            if d <= rem:
                chosen.append(d)
                rem -= d
        assert rem == 0
        return chosen
    
    # Reconstruct the solution
    chosen = []
    s = target
    while s > 0:
        prev_s, d = dp_parent[s]
        chosen.append(d)
        s = prev_s
    
    # Verify
    assert len(set(chosen)) == len(chosen), "divisors not distinct"
    assert sum(chosen) == target, f"sum {sum(chosen)} != {target}"
    assert all(n % d == 0 for d in chosen), "not all divisors of 5040"
    assert len(chosen) <= 256, "too many divisors"
    
    return chosen
# EVOLVE-BLOCK-END
