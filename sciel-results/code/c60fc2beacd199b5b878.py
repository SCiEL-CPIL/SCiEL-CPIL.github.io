def construct():
    # Generate all divisors of 5040
    n = 5040
    target = 4979
    divisors = [d for d in range(1, n+1) if n % d == 0]
    divisors.sort(reverse=True)  # not strictly necessary but might help
    
    # dp[s] = minimum number of divisors to achieve sum s
    # Initialize with infinity
    INF = float('inf')
    dp = {0: 0}
    # parent[s] = (prev_sum, divisor_used) for reconstruction
    parent = {0: (None, None)}
    
    # Process each divisor
    for d in divisors:
        # Iterate over current states in reverse to avoid using same divisor twice
        new_dp = dict(dp)
        new_parent = dict(parent)
        for s, cnt in list(dp.items()):
            new_s = s + d
            if new_s <= target and (new_s not in new_dp or cnt + 1 < new_dp[new_s]):
                new_dp[new_s] = cnt + 1
                new_parent[new_s] = (s, d)
        dp = new_dp
        parent = new_parent
    
    if target not in dp:
        raise ValueError("No solution found")
    
    # Reconstruct the solution
    chosen = []
    s = target
    while s != 0:
        prev_s, d = parent[s]
        chosen.append(d)
        s = prev_s
    
    return chosen
