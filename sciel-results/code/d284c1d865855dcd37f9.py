import time

def _get_divisors(n):
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return divisors

def construct():
    start_time = time.time()
    timeout = 50.0
    
    n = 5040
    target = 4979
    
    divisors = _get_divisors(n)
    N = len(divisors)
    
    # dp[s] = min count to reach sum s
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # parent[s] = (prev_sum, divisor_index) to reconstruct
    parent_sum = [-1] * (target + 1)
    parent_div_idx = [-1] * (target + 1)
    
    # Iterate through divisors
    # Sort divisors in descending order to potentially find a valid solution
    # faster or allow for better pruning if we were doing branch and bound,
    # but for standard 0/1 knapsack DP the order doesn't change the optimal
    # count, only the specific set of divisors chosen if there are ties.
    # However, sorting can sometimes help with cache locality or just be a
    # standard practice. The current program iterates in the order returned
    # by _get_divisors which is roughly sorted by size (small then large).
    # Let's keep the standard order or sort explicitly. 
    # Actually, the order of items in 0/1 knapsack DP does not affect the
    # optimality of the count, but it affects which specific subset is found
    # if there are multiple subsets with the same minimum count.
    # Since the goal is just to minimize count, any valid subset is fine.
    
    for i, d in enumerate(divisors):
        # Update dp backwards
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent_sum[s] = s - d
                parent_div_idx[s] = i
        
        if time.time() - start_time > timeout:
            break

    if dp[target] == INF:
        # Fallback: Greedy
        remainder = target
        chosen = []
        for d in sorted(divisors, reverse=True):
            if d <= remainder:
                chosen.append(d)
                remainder -= d
                if remainder == 0:
                    break
        return chosen

    # Reconstruct
    result = []
    s = target
    while s > 0:
        idx = parent_div_idx[s]
        if idx == -1:
            # Should not happen if dp[target] != INF, but safety check
            break
        result.append(divisors[idx])
        s = parent_sum[s]
        
    return result
