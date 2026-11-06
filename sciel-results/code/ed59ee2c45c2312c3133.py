def construct():
    n = 5040
    target = 4979
    
    # Generate all distinct positive divisors of n
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    
    # Filter out divisors larger than the target
    divisors = [d for d in divisors if d <= target]
    
    # DP approach to find minimum number of distinct divisors summing to target
    # dp[s] stores the minimum number of divisors needed to sum to s
    # parent[s] stores the last divisor added to reach s
    # prev_sum[s] stores the sum before adding the last divisor
    
    inf = float('inf')
    dp = [inf] * (target + 1)
    parent = [-1] * (target + 1)
    prev_sum = [-1] * (target + 1)
    
    dp[0] = 0
    
    # 0/1 Knapsack style: iterate items, then sums in reverse
    for d in divisors:
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = d
                prev_sum[s] = s - d
                
    if dp[target] == inf:
        # Fallback if no solution found (should not happen for this specific instance)
        # Greedy fallback
        remainder = target
        chosen = []
        for d in sorted(divisors, reverse=True):
            if d <= remainder:
                chosen.append(d)
                remainder -= d
        return chosen

    # Reconstruct the solution
    result = []
    s = target
    while s > 0:
        d = parent[s]
        result.append(d)
        s = prev_sum[s]
        
    return result
