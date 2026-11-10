import sys
import time

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
    
    # Sort divisors. Order doesn't strictly matter for DP result, 
    # but let's sort descending for potential early pruning if we were doing DFS.
    # For DP, any order works.
    divisors.sort(reverse=True)
    N = len(divisors)
    
    # DP table
    # dp[i][s] = min count using divisors[i:] to sum to s
    # We use a large number for infinity
    INF = 10**9
    # We only need current and next row to save memory, but we need choice table for backtracking.
    # So we store full dp and choice tables.
    
    # dp[i][s] for i in 0..N, s in 0..target
    # Dimensions: (N+1) x (target+1)
    dp = [[INF] * (target + 1) for _ in range(N + 1)]
    choice = [[0] * (target + 1) for _ in range(N)] # 0: skip, 1: take
    
    # Base case: no divisors left
    dp[N][0] = 0
    # dp[N][s > 0] is already INF
    
    # Fill DP table
    for i in range(N - 1, -1, -1):
        d = divisors[i]
        for s in range(target + 1):
            # Option 1: Skip divisor i
            best_val = dp[i + 1][s]
            best_choice = 0
            
            # Option 2: Take divisor i (if s >= d)
            if s >= d:
                take_val = 1 + dp[i + 1][s - d]
                if take_val < best_val:
                    best_val = take_val
                    best_choice = 1
            
            dp[i][s] = best_val
            choice[i][s] = best_choice
            
    # Check if solution exists
    if dp[0][target] == INF:
        # Should not happen for this specific instance, but handle gracefully
        return []
        
    # Backtrack to find the divisors
    result = []
    s = target
    for i in range(N):
        if choice[i][s] == 1:
            result.append(divisors[i])
            s -= divisors[i]
            
    return result
