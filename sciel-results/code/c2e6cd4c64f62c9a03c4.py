import sys

def construct():
    n = 5040
    target = 4979
    
    # Generate divisors
    divisors = [d for d in range(1, n + 1) if n % d == 0]
    num_divs = len(divisors)
    
    # DP setup
    # dp[i][s] = min number of divisors from divisors[0..i-1] to sum to s
    # We use a large number for infinity
    INF = 10**9
    # We can optimize space to 1D if we just want the min count, 
    # but we need to reconstruct the set. 
    # 2D is safer for reconstruction logic simplicity, or we can store parent pointers.
    # Given small size, 2D is fine.
    
    # dp[s] will store the min count.
    # To reconstruct, we need to know which divisor was used.
    # Let's use a 2D array for counts and a 2D array for backtracking.
    # take[i][s] = 1 if divisor i-1 was taken to reach sum s with optimal count for first i divisors.
    
    # Dimensions: (num_divs + 1) x (target + 1)
    dp = [[INF] * (target + 1) for _ in range(num_divs + 1)]
    take = [[0] * (target + 1) for _ in range(num_divs + 1)]
    
    dp[0][0] = 0
    
    for i in range(1, num_divs + 1):
        d = divisors[i-1]
        for s in range(target + 1):
            # Option 1: Don't take divisor i-1
            dp[i][s] = dp[i-1][s]
            
            # Option 2: Take divisor i-1
            if s >= d:
                if dp[i-1][s-d] + 1 < dp[i][s]:
                    dp[i][s] = dp[i-1][s-d] + 1
                    take[i][s] = 1
    
    # Check if solution exists
    if dp[num_divs][target] == INF:
        # Should not happen for practical numbers, but handle gracefully
        return []
        
    # Backtrack
    selected = []
    s = target
    for i in range(num_divs, 0, -1):
        if take[i][s] == 1:
            selected.append(divisors[i-1])
            s -= divisors[i-1]
            
    return selected
