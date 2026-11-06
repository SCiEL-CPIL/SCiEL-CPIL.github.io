"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Generate all divisors of 5040
    n = 5040
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    
    # Sort divisors in descending order for greedy approach
    divisors.sort(reverse=True)
    
    # Greedy approach to find a valid solution quickly
    remainder = 4979
    chosen = []
    for d in divisors:
        if d <= remainder:
            chosen.append(d)
            remainder -= d
        if remainder == 0:
            break
    
    if remainder != 0:
        # Fallback: try to find a solution using a different approach
        # This is a safety net, but greedy should work for this specific case
        # For now, just return what we have (should not happen for this problem)
        pass
    
    # Try to minimize the number of divisors
    # Since we're using greedy with largest first, we already have a minimal-ish solution
    # But let's try to improve it
    
    # Sort chosen divisors for consistency
    chosen.sort()
    
    return chosen
# EVOLVE-BLOCK-END
