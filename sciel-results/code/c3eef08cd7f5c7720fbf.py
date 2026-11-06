import time
from math import isqrt

def construct():
    deadline = time.time() + 85
    TARGET = 123457
    
    # Generate primes up to a reasonable limit using sieve
    def sieve(n):
        if n < 2:
            return []
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    # Generate primes up to 10000 to cover all possibilities
    primes_list = sieve(10000)
    
    def generate_smooth_numbers(limit, primes):
        """
        Generates all numbers <= limit that are composed of the given primes.
        Returns a set of such numbers.
        """
        smooths = set([1])
        frontier = [1]
        while frontier:
            x = frontier.pop()
            for p in primes:
                nx = x * p
                if nx > limit:
                    break
                if nx not in smooths:
                    smooths.add(nx)
                    frontier.append(nx)
        return smooths

    def exists_smooth_pair(primes):
        """Check if there exists a pair (a,b) with a+b=TARGET where both are smooth wrt primes."""
        smooths = generate_smooth_numbers(TARGET, primes)
        for a in smooths:
            b = TARGET - a
            if b > 0 and b in smooths:
                return (a, b)
        return None

    # Binary search on the smoothness bound B
    # We want to find the smallest B such that there exists a pair (a,b) with a+b=TARGET
    # where both a and b are B-smooth (all prime factors <= B)
    
    # First, find an upper bound where a solution definitely exists
    # B = TARGET-1 should work (a=1, b=TARGET-1, both trivially smooth)
    low_idx = 0
    high_idx = len(primes_list) - 1
    
    # Verify that high_idx works
    if time.time() > deadline - 10:
        return [1, TARGET - 1]
    
    high_primes = primes_list[:high_idx+1]
    result = exists_smooth_pair(high_primes)
    if result is None:
        # Shouldn't happen, but fallback
        return [1, TARGET - 1]
    
    # Now binary search for the smallest B that works
    best_pair = result
    best_B = primes_list[high_idx]
    
    while low_idx <= high_idx:
        if time.time() > deadline - 10:
            break
        
        mid_idx = (low_idx + high_idx) // 2
        mid_B = primes_list[mid_idx]
        
        # Generate primes up to mid_B
        mid_primes = primes_list[:mid_idx+1]
        
        result = exists_smooth_pair(mid_primes)
        
        if result is not None:
            best_pair = result
            best_B = mid_B
            high_idx = mid_idx - 1
        else:
            low_idx = mid_idx + 1
    
    if best_pair:
        return [best_pair[0], best_pair[1]]
    else:
        return [1, TARGET - 1]
