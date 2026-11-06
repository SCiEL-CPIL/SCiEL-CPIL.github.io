import math

def construct():
    n = 5040
    target = 4979
    
    # Generate all distinct positive divisors of n
    divisors = []
    for d in range(1, int(math.isqrt(n)) + 1):
        if n % d == 0:
            divisors.append(d)
            if d != n // d:
                divisors.append(n // d)
            
    # Filter out divisors larger than the target
    divisors = [d for d in divisors if d <= target]
    
    # Sort divisors for consistent backtracking (though not strictly required for min count)
    divisors.sort()
    N = len(divisors)
    
    # DP table: dp[i][s] = min number of divisors from first i divisors summing to s
    # We use a large number for infinity
    INF = 10**9
    dp = [[INF] * (target + 1) for _ in range(N + 1)]
    dp[0][0] = 0
    
    # Fill DP table
    for i in range(1, N + 1):
        d = divisors[i - 1]
        for s in range(target + 1):
            # Option 1: Don't use divisor d
            dp[i][s] = dp[i - 1][s]
            # Option 2: Use divisor d (if s >= d)
            if s >= d:
                if dp[i - 1][s - d] + 1 < dp[i][s]:
                    dp[i][s] = dp[i - 1][s - d] + 1
    
    # Check if solution exists
    if dp[N][target] == INF:
        # Fallback: should not happen for this specific instance
        # But let's return a valid set if possible, e.g., greedy
        remainder = target
        chosen = []
        for d in sorted(divisors, reverse=True):
            if d <= remainder:
                chosen.append(d)
                remainder -= d
                if remainder == 0:
                    break
        return chosen

    # Backtrack to find the actual divisors
    result = []
    s = target
    for i in range(N, 0, -1):
        # If the min count for (i, s) is the same as (i-1, s), we didn't use divisors[i-1]
        if dp[i][s] == dp[i - 1][s]:
            continue
        else:
            # We used divisors[i-1]
            d = divisors[i - 1]
            result.append(d)
            s -= d
    
    return result
