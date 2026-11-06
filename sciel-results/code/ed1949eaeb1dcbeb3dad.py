"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

def construct():
    start_time = time.time()
    deadline = start_time + 50  # safety margin before 90s kill
    
    n = 5040
    target = 4979
    
    # Generate all divisors of 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors in descending order for greedy-like initial solution,
    # but we'll use DP for exact minimum cardinality
    divisors.sort(reverse=True)
    
    # We need to find the minimum number of distinct divisors that sum to target
    # This is a subset sum problem with the additional constraint of minimizing count
    
    # DP approach: dp[s] = minimum number of divisors needed to sum to s
    # We process each divisor once (0/1 knapsack style)
    
    # Since target is 4979, we can use a DP array of size target+1
    # dp[s] = min count to achieve sum s, using divisors processed so far
    
    INF = float('inf')
    dp = [INF] * (target + 1)
    dp[0] = 0
    
    # Also track parent for reconstruction
    # parent[s] = (previous_sum, divisor_used)
    parent = [None] * (target + 1)
    
    # Process each divisor
    # To minimize count, we should try larger divisors first? 
    # Actually, for 0/1 knapsack, order doesn't matter for correctness,
    # but we need to be careful about the reconstruction
    
    # Let's use a different approach: 
    # dp[i][s] = min count using first i divisors to sum to s
    # But that's O(len(divisors) * target) which is 60 * 4979 ≈ 300k, very fast
    
    # Actually, let's use 1D DP with careful backtracking
    # We'll store for each sum, which divisor was last added
    
    # Better: use a dictionary or array where dp[s] = (min_count, last_divisor_index)
    # But we need to ensure each divisor is used at most once
    
    # Let's do it properly with 2D DP concept but 1D storage
    # We iterate divisors and update dp in reverse order to avoid reuse
    
    # dp[s] = minimum number of divisors to sum to s
    # We'll track the path separately
    
    # For reconstruction, let's store:
    # chosen[s] = list of divisors that sum to s with minimum count
    # But that could be memory intensive
    
    # Alternative: store prev[s] = (s - d, d) for the last step
    # But we need to ensure we're not reusing divisors
    
    # Let's use a different strategy: 
    # Since the number of divisors is small (60), we can use meet-in-the-middle
    # or just do a BFS/DFS with pruning
    
    # Actually, let's just do the DP properly:
    # dp[i][s] = min count using divisors[i:] to sum to s
    # This is O(60 * 4979) = ~300k states, very manageable
    
    # But for reconstruction, let's store the choice at each state
    
    # Let's use: dp[i][s] = min count to sum to s using divisors from index i onwards
    # choice[i][s] = True if we take divisors[i], False otherwise
    
    m = len(divisors)
    
    # Initialize
    # dp[i][s] for i from m-1 down to 0
    # Base case: dp[m][0] = 0, dp[m][s>0] = INF
    
    # Use arrays
    dp = [[INF] * (target + 1) for _ in range(m + 1)]
    choice = [[False] * (target + 1) for _ in range(m + 1)]
    
    dp[m][0] = 0
    
    for i in range(m - 1, -1, -1):
        d = divisors[i]
        for s in range(target + 1):
            # Option 1: don't take divisors[i]
            best = dp[i + 1][s]
            choice_val = False
            
            # Option 2: take divisors[i] if s >= d
            if s >= d and dp[i + 1][s - d] + 1 < best:
                best = dp[i + 1][s - d] + 1
                choice_val = True
            
            dp[i][s] = best
            choice[i][s] = choice_val
    
    # Check if solution exists
    if dp[0][target] == INF:
        # Fallback to greedy (should not happen for practical numbers)
        remainder = target
        chosen = []
        for d in divisors:
            if d <= remainder:
                chosen.append(d)
                remainder -= d
        assert remainder == 0, "No solution found"
        return chosen
    
    # Reconstruct
    chosen = []
    i = 0
    s = target
    while i < m:
        if choice[i][s]:
            chosen.append(divisors[i])
            s -= divisors[i]
        i += 1
    
    assert sum(chosen) == target, f"Sum mismatch: {sum(chosen)} != {target}"
    assert len(set(chosen)) == len(chosen), "Divisors not distinct"
    for d in chosen:
        assert n % d == 0, f"{d} is not a divisor of {n}"
    
    return chosen

# EVOLVE-BLOCK-END
