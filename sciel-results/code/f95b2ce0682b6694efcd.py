import time
from math import gcd

def construct():
    N = 512
    deadline = time.time() + 80

    # Find the integer M in [1, N] with the maximum number of divisors
    # The set of divisors of M is a valid clique.
    # Based on the task knowledge, this is the optimal strategy.
    
    max_div_count = 0
    best_M = 1
    best_divisors = [1]
    
    for M in range(1, N + 1):
        # Count divisors of M
        count = 0
        # Efficient divisor counting
        # Check i from 1 to sqrt(M)
        i = 1
        while i * i <= M:
            if M % i == 0:
                count += 1
                if i != M // i:
                    count += 1
            i += 1
        
        if count > max_div_count:
            max_div_count = count
            best_M = M
            # Store divisors if needed, or recompute at the end
            # We can just store M and compute divisors at the end
            
    # Compute divisors of best_M
    divisors = []
    i = 1
    while i * i <= best_M:
        if best_M % i == 0:
            divisors.append(i)
            if i != best_M // i:
                divisors.append(best_M // i)
        i += 1
    
    # Ensure non-empty
    if not divisors:
        divisors = [1]
        
    return divisors
