import time
import sys

def construct():
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    divisors.sort()
    N = len(divisors)
    
    # DP table: dp[i][s] = min number of divisors from first i divisors to sum to s
    # Initialize with infinity
    INF = float('inf')
    dp = [[INF] * (target + 1) for _ in range(N + 1)]
    dp[0][0] = 0
    
    for i in range(1, N + 1):
        d = divisors[i-1]
        prev_dp = dp[i-1]
        curr_dp = dp[i]
        # Copy previous row
        # curr_dp[:] = prev_dp[:] 
        # But we can update in place if we are careful? No, we need prev_dp values.
        # Actually, we can just iterate s.
        for s in range(target + 1):
            # Option 1: Don't use d
            curr_dp[s] = prev_dp[s]
            # Option 2: Use d
            if s >= d:
                if prev_dp[s - d] + 1 < curr_dp[s]:
                    curr_dp[s] = prev_dp[s - d] + 1
    
    min_count = dp[N][target]
    
    if min_count == INF:
        return []
        
    # Backtrack
    result = []
    s = target
    for i in range(N, 0, -1):
        d = divisors[i-1]
        if dp[i][s] == dp[i-1][s]:
            # d not used
            continue
        else:
            # d used
            result.append(d)
            s -= d
            
    return result

if __name__ == "__main__":
    res = construct()
    print(res)
    print(len(res))
    print(sum(res))
