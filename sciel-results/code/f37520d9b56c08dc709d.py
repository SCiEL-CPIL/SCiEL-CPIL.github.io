import time

def construct():
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            divisors.append(d)
    
    m = len(divisors)
    
    # DP to find minimum number of divisors summing to target
    # dp[i][s] = min count using first i divisors to sum to s
    INF = 10**9
    dp = [[INF] * (target + 1) for _ in range(m + 1)]
    choice = [[False] * (target + 1) for _ in range(m + 1)]
    
    dp[0][0] = 0
    
    # Optimize inner loop by local references
    for i in range(1, m + 1):
        d = divisors[i - 1]
        dp_i = dp[i]
        dp_prev = dp[i - 1]
        choice_i = choice[i]
        
        # Copy previous row (skip option)
        # We can optimize this by only copying up to target
        # But simple copy is fine for 5000 elements
        for s in range(target + 1):
            dp_i[s] = dp_prev[s]
            choice_i[s] = False
        
        # Take option
        # Iterate backwards? No, we are using 2D DP, so forward is fine as we read from prev row
        for s in range(d, target + 1):
            if dp_prev[s - d] + 1 < dp_i[s]:
                dp_i[s] = dp_prev[s - d] + 1
                choice_i[s] = True
    
    min_count = dp[m][target]
    
    if min_count == INF:
        raise ValueError("No solution found")
    
    # Backtrack
    chosen = []
    s = target
    for i in range(m, 0, -1):
        if choice[i][s]:
            chosen.append(divisors[i - 1])
            s -= divisors[i - 1]
    
    return chosen
