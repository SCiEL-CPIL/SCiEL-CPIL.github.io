"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    n = 5040
    target = 4979

    # Generate all distinct positive divisors of 5040
    divisors = []
    for d in range(1, n + 1):
        if n % d == 0:
            divisors.append(d)
    
    # Filter divisors that are larger than the target, as they cannot be part of the sum
    divisors = [d for d in divisors if d <= target]
    
    # DP approach to find the minimum number of distinct divisors summing to target
    # dp[s] will store the minimum number of divisors needed to sum to s
    # parent[s] will store the last divisor used to reach sum s, for reconstruction
    INF = float('inf')
    dp = [INF] * (target + 1)
    parent = [(-1, -1)] * (target + 1)  # (prev_sum, divisor_used)
    
    dp[0] = 0
    parent[0] = (-1, 0) # Start state
    
    # Iterate over each divisor
    for d in divisors:
        # Update dp in reverse to ensure each divisor is used at most once (0/1 knapsack)
        # We only need to update up to target
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                parent[s] = (s - d, d)
        
        # Check time limit, though this loop is very fast
        if time.time() > deadline:
            break

    if dp[target] == INF:
        # Fallback: This should not happen as 5040 is a practical number and 
        # 4979 is less than 5040, but just in case.
        # Greedy fallback
        remainder = target
        chosen = []
        for d in sorted(divisors, reverse=True):
            if d <= remainder:
                chosen.append(d)
                remainder -= d
            if remainder == 0:
                break
        if remainder == 0:
            return chosen
        return [1] # Should never reach here

    # Reconstruct the solution
    solution = []
    s = target
    while s > 0:
        prev_s, d = parent[s]
        if d == 0:
            break # Should not happen if dp[target] != INF
        solution.append(d)
        s = prev_s
    
    # Verify the solution
    if len(set(solution)) != len(solution):
        # Should not happen with 0/1 knapsack, but check just in case
        return []
    if sum(solution) != target:
        return []
    if any(5040 % d != 0 for d in solution):
        return []
    if len(solution) > 256:
        return []
        
    return solution
# EVOLVE-BLOCK-END
