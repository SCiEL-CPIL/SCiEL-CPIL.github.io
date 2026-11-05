"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import isqrt

def construct():
    N = 123457
    
    # Sieve for primes up to N
    def sieve(n):
        is_prime = [True] * (n + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, isqrt(n) + 1):
            if is_prime[i]:
                for j in range(i*i, n+1, i):
                    is_prime[j] = False
        return [i for i in range(2, n+1) if is_prime[i]]
    
    primes = sieve(N)
    
    # Incrementally build the set of B-smooth numbers
    # Start with 1
    smooth_set = {1}
    
    # Iterate through primes
    for idx, p in enumerate(primes):
        # Add multiples of p to the existing smooth numbers
        # We need to be careful to use a snapshot of the current set
        # because we are adding new numbers that might be multiplied by p again?
        # No, the standard algorithm is:
        # For each prime p, for each existing smooth number s, add s*p, s*p^2, ...
        # This works if we process primes in order.
        
        new_adds = set()
        for s in list(smooth_set):
            prod = s
            while prod <= N // p:
                prod *= p
                if prod > N:
                    break
                if prod not in smooth_set:
                    new_adds.add(prod)
        
        smooth_set.update(new_adds)
        
        # Check for a pair summing to N
        # We can optimize by only checking a <= N//2 to avoid duplicates?
        # But we just need one pair.
        for a in smooth_set:
            if a > N:
                continue
            b = N - a
            if b < 1:
                continue
            if b in smooth_set:
                # Found a pair with max prime factor <= p
                return [a, b]
    
    # Fallback
    return [1, 123456]
# EVOLVE-BLOCK-END
