import math
import time

def construct():
    n = 5040
    target = 4979
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    # Generate divisors
    divisors = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors in descending order for potential early pruning
    divisors.sort(reverse=True)
    
    # DP for 0/1 Knapsack minimizing number of items
    # dp[s] = min number of divisors to sum to s
    # choice[s] = the divisor that was added to reach s optimally
    
    dp = [float('inf')] * (target + 1)
    choice = [0] * (target + 1)
    dp[0] = 0
    
    for d in divisors:
        if d > target:
            continue
        # Check time limit
        if time.time() > deadline:
            break
        # Iterate downwards to ensure each divisor is used at most once
        for s in range(target, d - 1, -1):
            if dp[s - d] + 1 < dp[s]:
                dp[s] = dp[s - d] + 1
                choice[s] = d
                
    # Check if solution exists
    if dp[target] == float('inf'):
        # Should not happen for this specific instance
        return []
        
    # Reconstruct the solution
    result = []
    current_sum = target
    while current_sum > 0:
        d = choice[current_sum]
        if d == 0:
            # This would imply a bug or no solution path, 
            # but dp[target] was finite, so this shouldn't happen
            break
        result.append(d)
        current_sum -= d
        
    return result
