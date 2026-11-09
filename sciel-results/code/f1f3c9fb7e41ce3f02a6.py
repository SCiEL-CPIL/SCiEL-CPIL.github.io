"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Get all divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort()
    divisors.reverse()  # largest first for potential early pruning
    
    num_divs = len(divisors)
    
    # 0/1 Knapsack DP: dp[s] = min number of divisors to achieve sum s
    INF = float('inf')
    dp_count = [INF] * (target + 1)
    # For reconstruction, store the previous state info
    # parent[s] = (prev_sum, divisor_index) that led to optimal dp[s]
    parent_sum = [-1] * (target + 1)
    parent_div = [-1] * (target + 1)
    
    dp_count[0] = 0
    
    for idx, d in enumerate(divisors):
        if d > target:
            continue
        # Process in reverse to ensure each divisor used at most once
        for s in range(target, d - 1, -1):
            prev = s - d
            if dp_count[prev] + 1 < dp_count[s]:
                dp_count[s] = dp_count[prev] + 1
                parent_sum[s] = prev
                parent_div[s] = idx
    
    if dp_count[target] == INF:
        # Fallback: greedy
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
        return chosen
    
    # Reconstruct the solution
    chosen = []
    s = target
    while s > 0:
        d_idx = parent_div[s]
        if d_idx == -1:
            break
        chosen.append(divisors[d_idx])
        s = parent_sum[s]
    
    # Verify
    assert sum(chosen) == target, f"Sum check failed: {sum(chosen)} != {target}"
    assert len(set(chosen)) == len(chosen), "Duplicates found"
    for d in chosen:
        assert n % d == 0, f"{d} is not a divisor of {n}"
    
    return chosen
# EVOLVE-BLOCK-END
