import math

def construct():
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors (optional, but good for consistency)
    divisors.sort()
    N = len(divisors)
    
    # DP table: dp[i][s] = min number of divisors from divisors[0..i-1] to sum to s
    # Use a large number for infinity
    INF = float('inf')
    dp = [[INF] * (target + 1) for _ in range(N + 1)]
    
    # Base case: sum 0 requires 0 divisors
    dp[0][0] = 0
    
    # Fill DP table
    for i in range(1, N + 1):
        d = divisors[i-1]
        for s in range(target + 1):
            # Option 1: Don't use divisor d
            dp[i][s] = dp[i-1][s]
            
            # Option 2: Use divisor d
            if s >= d and dp[i-1][s-d] != INF:
                # If using d results in a smaller count
                if dp[i-1][s-d] + 1 < dp[i][s]:
                    dp[i][s] = dp[i-1][s-d] + 1
    
    # Check if solution exists
    if dp[N][target] == INF:
        return [] # Should not happen for this task
    
    # Backtrack to find the divisors
    selected = []
    s = target
    for i in range(N, 0, -1):
        d = divisors[i-1]
        # If the state didn't change from previous row, we didn't use this divisor
        if dp[i][s] == dp[i-1][s]:
            continue
        else:
            # We used this divisor
            selected.append(d)
            s -= d
    
    return selected
