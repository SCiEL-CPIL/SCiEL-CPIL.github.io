"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
from itertools import combinations

# EVOLVE-BLOCK-START

def construct():
    deadline = time.time() + 145.0  # safety margin
    
    # Product of first 15 primes
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    n_squarefree = 1
    for p in primes:
        n_squarefree *= p
    # n_squarefree = 614889782588491410 <= 10^18
    
    # Generate all divisors of n_squarefree
    # For each subset of primes, the product is a divisor
    divs = [1]
    for p in primes:
        divs += [d * p for d in divs]
    divs.sort()
    
    # Compute tau_perp
    score_squarefree = 0
    for i in range(len(divs) - 1):
        if math.gcd(divs[i], divs[i+1]) == 1:
            score_squarefree += 1
    
    best_n = n_squarefree
    best_score = score_squarefree
    
    # Try to see if replacing a prime with a prime power of a smaller prime helps?
    # Or just check a few nearby squarefree numbers?
    # Actually, let's check if any prime power of a prime in the set, replacing that prime,
    # yields a higher score.
    # e.g. replace 47 with 2^k? No, 2 is already there.
    # Replace 47 with 3^k?
    
    # Let's try replacing the largest prime p_max with a prime power p^k such that
    # p^k < p_max and the new number is <= 10^18.
    # Actually, if we replace p with p^k, the number of divisors increases.
    # But the structure of coprime adjacency might suffer.
    
    # Let's just verify the squarefree result is robust.
    # If we have time, we can try to optimize further.
    
    # For now, return the squarefree product.
    return best_n

# EVOLVE-BLOCK-END
