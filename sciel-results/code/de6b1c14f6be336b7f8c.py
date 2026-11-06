import math
import time

def construct():
    deadline = time.time() + 145.0
    N = 512
    
    # Precompute prime factors for all numbers 1..511
    # prime_factors[a] is a set of prime factors of a
    prime_factors = [set() for _ in range(N)]
    
    # Sieve-like approach to find prime factors
    for i in range(2, N):
        if not prime_factors[i]: # i is prime
            for j in range(i, N, i):
                prime_factors[j].add(i)
    
    # 1 has no prime factors
    prime_factors[1] = set()
    
    # Sort candidates by value descending (a descending)
    # Value is 1/(512-a)
    candidates = list(range(511, 0, -1))
    
    used_primes = set()
    selected = []
    
    for a in candidates:
        if time.time() > deadline:
            break
            
        pf = prime_factors[a]
        
        # Check if coprime with all selected (equivalent to disjoint prime factors)
        if not (pf & used_primes):
            selected.append(a)
            used_primes.update(pf)
            
    # Ensure 1 is included (it's always coprime and adds positive value)
    # 1 is in candidates, and prime_factors[1] is empty, so it will be picked 
    # at the end of the loop if not already picked.
    # Since we iterate 511 down to 1, 1 is the last one.
    # It will be added if no conflict (which is never).
    
    return selected
