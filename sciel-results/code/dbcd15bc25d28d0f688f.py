import math

# Pre-compute primes up to 10^6 using Sieve of Eratosthenes
LIMIT = 10**6
is_prime_sieve = [True] * (LIMIT + 1)
is_prime_sieve[0] = is_prime_sieve[1] = False
for i in range(2, int(LIMIT**0.5) + 1):
    if is_prime_sieve[i]:
        for j in range(i*i, LIMIT + 1, i):
            is_prime_sieve[j] = False

primes = [i for i, prime in enumerate(is_prime_sieve) if prime]

def factor(n):
    """Factor n using trial division by precomputed primes."""
    if n == 1:
        return []
    factors = []
    temp = n
    for p in primes:
        if p * p > temp:
            break
        while temp % p == 0:
            factors.append(p)
            temp //= p
    if temp > 1:
        factors.append(temp)
    return factors

def phi(n):
    if n == 1:
        return 1
    factors = factor(n)
    unique_primes = set(factors)
    result = n
    for p in unique_primes:
        result -= result // p
    return result

def construct():
    # We will try to find the maximum k.
    # Based on literature/heuristics, k=200 is very unlikely.
    # We'll step down from 100 to 1.
    # For each k, we will check a set of candidate starts.
    
    # To improve efficiency, we can check if the numbers in the block
    # have "similar" phi values. But exact check is required.
    
    # Candidate starts strategy:
    # 1. Near 10^12
    # 2. Near 10^11
    # 3. Random-ish deterministic spread
    
    max_k = 100
    for k in range(max_k, 0, -1):
        # Generate candidate starts
        candidates = set()
        max_start = 10**12 - k + 1
        
        # Add points near the top
        for offset in [0, 1000, 10000, 100000, 1000000]:
            s = max_start - offset
            if s >= 1:
                candidates.add(s)
        
        # Add points in other ranges
        for base in [10**11, 10**10, 10**9]:
            if base <= max_start:
                candidates.add(base)
                candidates.add(base + 1000)
                candidates.add(base + 10000)
        
        # Add some "random" looking but deterministic points
        # Using a simple LCG or just fixed offsets
        for i in range(20):
            s = (max_start // (i + 2)) 
            if s >= 1 and s <= max_start:
                candidates.add(s)
            
        # Sort candidates to try larger ones first (often better for distinctness? Not necessarily, but good to try high range)
        valid_candidates = [s for s in candidates if 1 <= s <= max_start]
        valid_candidates.sort(reverse=True)
        
        for s in valid_candidates:
            seen = set()
            valid = True
            for i in range(k):
                val = phi(s + i)
                if val in seen:
                    valid = False
                    break
                seen.add(val)
            
            if valid:
                return {'start': s, 'length': k}
        
        # If we fail for this k, we might want to try more candidates?
        # Given the time limit, we just drop k.
        
    return {'start': 1, 'length': 1}
