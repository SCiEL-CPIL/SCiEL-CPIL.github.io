def construct():
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
        i += 1
    
    # dp[s] = min number of divisors to sum to s
    INF = 10**9
    dp = [INF] * (target + 1)
    dp[0] = 0
    parent = [None] * (target + 1)
    
    # 0/1 Knapsack DP
    for d in divisors:
        # Iterate backwards to ensure each divisor is used at most once
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = d
                
    if dp[target] == INF:
        raise ValueError("No solution found")
        
    # Reconstruct
    chosen = []
    s = target
    while s > 0:
        d = parent[s]
        if d is None:
            raise ValueError("Reconstruction failed")
        chosen.append(d)
        s -= d
        
    return chosen
