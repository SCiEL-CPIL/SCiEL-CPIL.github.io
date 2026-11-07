import time
from math import isqrt

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a list of distinct integers in 2..256 such that for every three 
    distinct members a, b, c, a does not divide b*c.
    
    Mathematical Analysis:
    1. The set of all primes in [2, 256] is a valid set.
       Proof: If a, b, c are distinct primes, a does not divide b and a does not divide c.
       Thus, the exponent of prime a in the factorization of b*c is 0, while in a it is 1.
       So a ∤ b*c.
    2. Can we add any composite number?
       Let S be a valid set. If S contains a prime p, it cannot contain any other multiple of p.
       Proof: If p, x, y are distinct elements of S and p|x, then for the triple (a=p, b=x, c=y),
       p divides x*y. This violates the condition.
       Therefore, for any prime p, at most one element in S can be a multiple of p.
       If that element is p itself, we use the prime.
       If that element is a composite multiple c, we "waste" the prime p (since p cannot be in S).
       Replacing c with p maintains validity (as shown above) and keeps cardinality same.
       Thus, the maximum cardinality is achieved by selecting one representative for each prime p <= 256.
       The primes themselves are the simplest representatives.
       There are 54 primes in [2, 256].
       
    Therefore, the optimal solution is the list of all primes in [2, 256].
    """
    # Sieve of Eratosthenes to find all primes up to 256
    limit = 256
    is_prime = [True] * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, isqrt(limit) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    
    primes = [i for i in range(2, limit + 1) if is_prime[i]]
    
    return primes
# EVOLVE-BLOCK-END
