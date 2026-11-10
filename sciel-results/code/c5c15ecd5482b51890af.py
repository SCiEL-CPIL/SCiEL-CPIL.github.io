import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = 5040
    target = 4979
    
    # 1. Generate divisors
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    divisors.sort()
    N = len(divisors)
    
    # 2. DP with NumPy
    INF = 1000
    dp = np.full((N + 1, target + 1), INF, dtype=np.int32)
    dp[0, 0] = 0
    
    for i in range(1, N + 1):
        d = divisors[i-1]
        prev_row = dp[i-1]
        curr_row = dp[i]
        
        curr_row[:] = prev_row
        
        if d <= target:
            candidates = prev_row[:-d] + 1
            np.minimum(curr_row[d:], candidates, out=curr_row[d:])
        
        if time.time() > deadline:
            break
    
    # Check if solution exists
    if dp[N, target] >= INF:
        # Fallback greedy
        remainder = target
        chosen = []
        for d in reversed(divisors):
            if d <= remainder:
                chosen.append(d)
                remainder -= d
                if remainder == 0:
                    break
        return chosen
    
    min_count = int(dp[N, target])
    
    # 3. Backtrack
    chosen = []
    s = target
    for i in range(N, 0, -1):
        if time.time() > deadline:
            break
            
        d = divisors[i-1]
        
        if s >= d:
            # Prefer not picking if it leads to same count (to keep numbers smaller, 
            # but actually we just need any valid solution with min count)
            # Pick d if it's part of an optimal solution
            if dp[i][s] == dp[i-1][s-d] + 1:
                chosen.append(d)
                s -= d
            # else: didn't pick d
        # else: s < d, can't pick
    
    # Verify
    if sum(chosen) != target:
        # Fallback: reconstruct more carefully
        chosen = []
        s = target
        for i in range(N, 0, -1):
            d = divisors[i-1]
            if s >= d and dp[i][s] == dp[i-1][s-d] + 1:
                chosen.append(d)
                s -= d
        if sum(chosen) != target:
            # Last resort greedy
            remainder = target
            chosen = []
            for d in reversed(divisors):
                if d <= remainder:
                    chosen.append(d)
                    remainder -= d
                    if remainder == 0:
                        break
    
    return chosen
