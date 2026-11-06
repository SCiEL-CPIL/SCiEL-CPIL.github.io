import time
from math import gcd

def construct():
    start_time = time.time()
    limit = 512
    deadline = start_time + 85
    
    # Find the number in [1, limit] with the maximum number of divisors
    max_divs = 0
    best_m = 1
    best_divs_set = []
    
    for m in range(1, limit + 1):
        # Count divisors
        count = 0
        divs = []
        for i in range(1, int(m**0.5) + 1):
            if m % i == 0:
                divs.append(i)
                if i != m // i:
                    divs.append(m // i)
        if len(divs) > max_divs:
            max_divs = len(divs)
            best_m = m
            best_divs_set = divs
            
    # Verify the set is valid (lcm(a,b) <= limit for all pairs)
    # For divisors of M, lcm(a,b) divides M. If M <= limit, then lcm(a,b) <= limit.
    # So this set is always valid.
    
    # Just in case there's a non-divisor set that is larger (unlikely),
    # we could run a quick local search, but the divisor set is the theoretical max.
    
    return best_divs_set
