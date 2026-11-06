"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START

def _sieve(n):
    """Generate all primes up to n using Sieve of Eratosthenes."""
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    return [i for i in range(2, n + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    primes = _sieve(1000)
    
    # To maximize the number of primes we can use (and thus the length L 
    # assuming p > L), we should use the primes with the smallest 
    # reciprocal costs. This means the LARGEST primes.
    primes_desc = sorted(primes, reverse=True)
    
    budget = Fraction(1)
    selected_primes = []
    
    for p in primes_desc:
        if time.time() > deadline:
            break
            
        cost = Fraction(1, p)
        if budget >= cost:
            budget -= cost
            selected_primes.append(p)
        else:
            # Since we are iterating from largest (cheapest) to smallest (most expensive),
            # if we can't afford this prime, we can't afford any subsequent ones.
            break
            
    # Construct the classes.
    # We have len(selected_primes) primes.
    # We assign the i-th prime (0-indexed) to cover the integer i+1.
    # Since all selected primes are likely > len(selected_primes),
    # the residue (i+1) % p is just i+1.
    # This ensures that integer i+1 is covered.
    # Because p > L, it covers no other integers in 1..L.
    # Thus, 1..L are all covered.
    
    classes = []
    for i, p in enumerate(selected_primes):
        r = (i + 1) % p
        classes.append([p, r])
        
    return classes

# EVOLVE-BLOCK-END
