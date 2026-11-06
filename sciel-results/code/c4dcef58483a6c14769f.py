import time

def construct():
    start_time = time.time()
    deadline = start_time + 50.0
    
    n = 5040
    target = 4979
    
    # Generate all positive divisors of 5040
    divs = []
    i = 1
    while i * i <= n:
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
        i += 1
        
    m = len(divs)
    
    # dp[i][s] = minimum number of divisors from the first i divisors 
    # to achieve sum s. INF if impossible.
    INF = 10**9
    # Using a list of lists for simplicity, 60 * 4980 is small
    # dp[i] corresponds to considering divisors[0] ... divisors[i-1]
    # Size: (m+1) x (target+1)
    dp = [[INF] * (target + 1) for _ in range(m + 1)]
    dp[0][0] = 0
    
    # Fill the DP table
    for i in range(1, m + 1):
        d = divs[i - 1]
        # Copy previous row
        # We can optimize memory by using 1D if we don't need backtracking, 
        # but we DO need backtracking to find the actual divisors.
        # So 2D is safer for reconstruction logic clarity, or we store parent pointers.
        # 2D is fine.
        for s in range(target + 1):
            # Option 1: Don't use divisor i-1
            dp[i][s] = dp[i - 1][s]
            
            # Option 2: Use divisor i-1
            if s >= d:
                if dp[i - 1][s - d] + 1 < dp[i][s]:
                    dp[i][s] = dp[i - 1][s - d] + 1
    
    # Check if solution exists
    if dp[m][target] == INF:
        return []
    
    # Backtrack to find the divisors
    result = []
    s = target
    for i in range(m, 0, -1):
        # Check if divisor i-1 was used
        if dp[i][s] != dp[i - 1][s]:
            result.append(divs[i - 1])
            s -= divs[i - 1]
        if s == 0:
            break
            
    return result
