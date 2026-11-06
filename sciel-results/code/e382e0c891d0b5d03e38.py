def construct():
    import time
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    n = 5040
    target = 4979

    # 1. Generate all distinct positive divisors of n
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort for deterministic behavior
    divisors.sort()
    N = len(divisors)
    
    # 2. Dynamic Programming
    # dp[i][s] = minimum number of divisors from divisors[i:] needed to sum to s
    # We use a large number to represent infinity
    INF = 10000
    
    # Optimization: Use 1D arrays for dp and take if possible?
    # No, we need to backtrack, so we need the history.
    # However, we can store the choice.
    
    # Memory check: 60 * 4980 integers.
    # In Python, this is a list of lists.
    # To be safe against memory limits (unlikely here) or speed, we can use array module or just lists.
    # Lists are fine.
    
    dp = [[INF] * (target + 1) for _ in range(N + 1)]
    take = [[False] * (target + 1) for _ in range(N)]
    
    dp[N][0] = 0
    
    for i in range(N - 1, -1, -1):
        d = divisors[i]
        dp_row = dp[i]
        dp_next_row = dp[i+1]
        take_row = take[i]
        
        for s in range(target + 1):
            # Option 1: Skip divisors[i]
            best_val = dp_next_row[s]
            best_take = False
            
            # Option 2: Take divisors[i]
            if s >= d:
                val = dp_next_row[s - d] + 1
                if val < best_val:
                    best_val = val
                    best_take = True
            
            dp_row[s] = best_val
            take_row[s] = best_take

    # Check for timeout (unlikely to be needed, but good practice per directive)
    if time.time() > deadline:
        # If we are here, DP is done. If DP wasn't done, we'd be stuck in the loop.
        # Since the loop is fast, this check is just a formality.
        pass

    # 3. Reconstruction
    if dp[0][target] == INF:
        # No solution found (should not happen for this instance)
        return []
        
    result = []
    s = target
    for i in range(N):
        if take[i][s]:
            result.append(divisors[i])
            s -= divisors[i]
            
    return result
