def construct():
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    num_divs = len(divisors)
    
    INF = float('inf')
    # dp[i][r] = min number of divisors from divisors[i:] summing to r
    # We use a list of lists. Size (num_divs+1) x (target+1)
    dp = [[INF] * (target + 1) for _ in range(num_divs + 1)]
    dp[num_divs][0] = 0
    
    # Fill DP table backwards
    for i in range(num_divs - 1, -1, -1):
        d = divisors[i]
        for r in range(target + 1):
            # Option 1: Skip divisor i
            dp[i][r] = dp[i + 1][r]
            # Option 2: Take divisor i
            if r >= d:
                prev = dp[i + 1][r - d]
                if prev != INF:
                    take_val = 1 + prev
                    if take_val < dp[i][r]:
                        dp[i][r] = take_val
                        
    if dp[0][target] == INF:
        return []
        
    # Backtrack to find the set of divisors
    selected = []
    r = target
    for i in range(num_divs):
        d = divisors[i]
        # Check if taking d is part of the optimal solution
        take_val = INF
        if r >= d and dp[i + 1][r - d] != INF:
            take_val = 1 + dp[i + 1][r - d]
            
        if take_val == dp[i][r]:
            selected.append(d)
            r -= d
            
    return selected
