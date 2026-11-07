"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
from math import isqrt

def _sieve_primes(n):
    """Return list of primes up to n."""
    sieve = [True] * (n + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            for j in range(i * i, n + 1, i):
                sieve[j] = False
    return [i for i in range(2, n + 1) if sieve[i]]

def _factorize(n, primes):
    """Return dict of prime->exponent for n."""
    factors = {}
    for p in primes:
        if p * p > n:
            break
        if n % p == 0:
            exp = 0
            while n % p == 0:
                n //= p
                exp += 1
            factors[p] = exp
    if n > 1:
        factors[n] = 1
    return factors

def construct():
    n = 256
    primes = _sieve_primes(n)
    
    # Compute prime factor dicts for each number in 2..n
    prime_factors = {}
    for i in range(2, n + 1):
        prime_factors[i] = _factorize(i, primes)
    
    def can_add(a, S_list):
        """Check if a can be added to list S_list without violating the condition."""
        S = S_list
        m = len(S)
        # Check: a ∤ b*c for all distinct b,c in S
        for i in range(m):
            for j in range(i + 1, m):
                b = S[i]
                c = S[j]
                if (b * c) % a == 0:
                    return False
        # Check: for each b in S, b ∤ a*c for all c in S, c != b
        for i in range(m):
            b = S[i]
            for j in range(m):
                if j == i:
                    continue
                c = S[j]
                if (a * c) % b == 0:
                    return False
        return True

    candidates = list(range(2, n + 1))
    
    # Ordering 1: Largest prime factor descending, then fewer prime factors
    def sort_key_lpf(x):
        pf = prime_factors[x]
        largest_pf = max(pf.keys()) if pf else 0
        num_pf = len(pf)
        return (-largest_pf, num_pf, -x)
    
    c1 = sorted(candidates, key=sort_key_lpf)
    S1 = []
    for a in c1:
        if can_add(a, S1):
            S1.append(a)
    
    # Ordering 2: More prime factors first, then largest prime factor
    def sort_key_npf(x):
        pf = prime_factors[x]
        largest_pf = max(pf.keys()) if pf else 0
        num_pf = len(pf)
        return (-num_pf, -largest_pf, -x)
    
    c2 = sorted(candidates, key=sort_key_npf)
    S2 = []
    for a in c2:
        if can_add(a, S2):
            S2.append(a)
    
    # Ordering 3: Descending value
    c3 = sorted(candidates, key=lambda x: -x)
    S3 = []
    for a in c3:
        if can_add(a, S3):
            S3.append(a)
    
    # Ordering 4: Primes first (by value descending), then composites by largest prime factor
    def sort_key_prime_first(x):
        pf = prime_factors[x]
        is_prime = len(pf) == 1
        largest_pf = max(pf.keys()) if pf else 0
        return (0 if is_prime else 1, -largest_pf, -x)
    
    c4 = sorted(candidates, key=sort_key_prime_first)
    S4 = []
    for a in c4:
        if can_add(a, S4):
            S4.append(a)
    
    # Pick the best
    best_S = max([S1, S2, S3, S4], key=len)
    
    return list(best_S)
# EVOLVE-BLOCK-END
