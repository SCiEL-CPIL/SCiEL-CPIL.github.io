"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _get_divisors(n):
    divs = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs, reverse=True)

def construct():
    n = 5040
    target = 4979
    divisors = _get_divisors(n)
    # Filter divisors > target as they cannot be part of the sum
    divisors = [d for d in divisors if d <= target]
    
    # dp[sum] = (min_count, tuple_of_divisors)
    # Initialize with sum 0
    dp = {0: (0, ())}
    
    # Iterate through each divisor
    for d in divisors:
        # Take a snapshot of the current dp state to avoid using 'd' multiple times
        # in the same iteration step (0/1 knapsack constraint)
        current_items = list(dp.items())
        
        for s, (count, subset) in current_items:
            new_s = s + d
            if new_s > target:
                continue
            
            new_count = count + 1
            
            # Check if this new combination improves the known best for new_s
            if new_s in dp:
                if new_count < dp[new_s][0]:
                    dp[new_s] = (new_count, subset + (d,))
            else:
                dp[new_s] = (new_count, subset + (d,))
                
    if target in dp:
        return list(dp[target][1])
    
    # Fallback: return empty list or raise error if not found
    # For this specific instance, a solution is guaranteed to exist.
    return []
# EVOLVE-BLOCK-END
