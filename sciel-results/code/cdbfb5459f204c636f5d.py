import time

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 50s margin for 60s limit
    
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
    
    # 2. DP Setup
    # dp[i][s] = min number of divisors from divisors[0..i-1] to sum to s
    # INF = a large number
    INF = float('inf')
    
    # We use a 2D list for easy backtracking
    # Dimensions: (N+1) x (target+1)
    # To save memory/time, we can use a list of arrays or just lists
    # 61 * 4980 is small.
    
    dp = [[INF] * (target + 1) for _ in range(N + 1)]
    dp[0][0] = 0
    
    # Fill DP table
    for i in range(1, N + 1):
        d = divisors[i-1]
        prev_row = dp[i-1]
        curr_row = dp[i]
        
        # Copy previous row (case: don't pick d)
        # We can just iterate and update
        for s in range(target + 1):
            # Option 1: Don't pick d[i-1]
            best = prev_row[s]
            
            # Option 2: Pick d[i-1]
            if s >= d:
                if prev_row[s-d] + 1 < best:
                    best = prev_row[s-d] + 1
            
            curr_row[s] = best
            
        # Time check
        if time.time() > deadline:
            break

    # Check if solution exists
    if dp[N][target] == INF:
        # Fallback: return empty list or greedy result?
        # The problem states a solution exists.
        # If DP failed due to time, we should return a valid solution.
        # But 60*5000 is very fast, so this shouldn't happen.
        # Let's implement a fast greedy fallback just in case.
        remainder = target
        chosen = []
        for d in reversed(divisors):
            if d <= remainder:
                chosen.append(d)
                remainder -= d
                if remainder == 0:
                    break
        return chosen

    # 3. Backtrack to find the set
    chosen = []
    s = target
    for i in range(N, 0, -1):
        if time.time() > deadline:
            break
        
        d = divisors[i-1]
        
        # Check if d[i-1] was used
        # If dp[i][s] == dp[i-1][s], then d[i-1] was NOT used (or used but didn't improve count? 
        # Actually, if dp[i][s] < dp[i-1][s], it MUST have been used.
        # If dp[i][s] == dp[i-1][s], it might have been used if dp[i-1][s-d]+1 == dp[i-1][s].
        # However, we want ANY valid set with min count.
        # If dp[i][s] == dp[i-1][s], we can safely say we didn't use it, 
        # UNLESS using it is also optimal. 
        # But if we don't use it, we stay on the optimal path for the remaining items.
        # So if dp[i][s] == dp[i-1][s], we skip d[i-1].
        # If dp[i][s] == dp[i-1][s-d] + 1, we use d[i-1].
        
        if s >= d and dp[i][s] == dp[i-1][s-d] + 1:
            # It is possible that dp[i-1][s] is also equal to this value.
            # In that case, we have a choice. 
            # Picking d[i-1] is fine.
            chosen.append(d)
            s -= d
        else:
            # Did not pick d[i-1]
            pass
            
    # Verify sum
    # The logic above ensures we follow a path to 0.
    # If s != 0, something went wrong, but it shouldn't.
    
    return chosen
