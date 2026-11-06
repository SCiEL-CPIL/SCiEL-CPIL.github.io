import time

def construct():
    start_time = time.time()
    deadline = 90.0  # Wall-clock limit
    
    n = 5040
    target = 4979
    
    # 1. Get all distinct divisors of n
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors, largest first might help if we were doing greedy, 
    # but for DP it doesn't matter much. Let's just sort them.
    divisors.sort()
    N = len(divisors)
    
    # 2. Dynamic Programming
    # dp[i][s] = minimum number of divisors from divisors[0..i-1] that sum to s
    # We use a 2D list for simplicity and to allow backtracking easily.
    # Dimensions: (N+1) x (target+1)
    # N is ~60, target is ~5000. 60*5000 = 300k entries. Very fast.
    
    INF = float('inf')
    dp = [[INF] * (target + 1) for _ in range(N + 1)]
    # take[i][s] = True if divisor divisors[i-1] is used in the optimal solution for dp[i][s]
    take = [[False] * (target + 1) for _ in range(N + 1)]
    
    # Base case: sum 0 requires 0 items
    dp[0][0] = 0
    
    # Fill DP table
    for i in range(1, N + 1):
        d = divisors[i-1]
        for s in range(target + 1):
            # Option 1: Don't take divisor d
            best_count = dp[i-1][s]
            best_take = False
            
            # Option 2: Take divisor d (if s >= d)
            if s >= d:
                prev_count = dp[i-1][s-d]
                if prev_count + 1 < best_count:
                    best_count = prev_count + 1
                    best_take = True
            
            dp[i][s] = best_count
            take[i][s] = best_take
            
            # Time check (unlikely to be needed for this size, but good practice per directive)
            if time.time() > deadline - 5.0:
                 # If we run out of time, we should return a valid solution if we have one,
                 # or fall back to greedy. But this loop is tiny.
                 pass

    # 3. Check if solution exists
    if dp[N][target] == INF:
        # Fallback: Greedy (should not happen for practical numbers usually, but safety)
        remainder = target
        chosen = []
        # Try largest divisors first
        for d in sorted(divisors, reverse=True):
            if d <= remainder:
                chosen.append(d)
                remainder -= d
        if remainder == 0:
            return chosen
        else:
            # If even greedy fails, return empty or raise? Task implies solution exists.
            # The prompt says "Exact membership and sum checks". 
            # If no solution, we can't return a valid list. 
            # But 4979 is likely representable.
            return []

    # 4. Backtrack to find the divisors
    chosen = []
    s = target
    for i in range(N, 0, -1):
        if take[i][s]:
            chosen.append(divisors[i-1])
            s -= divisors[i-1]
    
    # The chosen list contains the divisors in reverse order of selection (largest index first).
    # Order doesn't matter for the set, but let's return it as is or sorted.
    # The prompt asks for "at most 256 distinct positive divisors".
    return chosen

# EVOLVE-BLOCK-END
