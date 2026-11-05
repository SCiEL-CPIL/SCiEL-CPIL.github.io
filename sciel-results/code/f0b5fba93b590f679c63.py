"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import combinations, product as iproduct

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin for 150s limit
    
    # 1. Generate primes up to a sufficient limit
    # The product of the first 15 primes is ~6.15e17. 
    # The 16th prime is 53. 6.15e17 * 53 > 10^18.
    # So we can have at most 15 distinct prime factors.
    # We need primes up to maybe 100 or so to be safe, but 100 is small.
    # Let's generate primes up to 200 just in case we want to skip some small ones and use larger ones?
    # No, to maximize the count of distinct primes, we want the smallest ones.
    # To maximize the score, we likely want the most distinct primes possible.
    
    def sieve(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    primes = sieve(200)
    
    # 2. Define scoring function
    # tau_perp(n) = number of adjacent pairs (d_i, d_{i+1}) in sorted divisors of n with gcd(d_i, d_{i+1}) == 1.
    
    def get_divisors(n):
        divs = []
        i = 1
        while i * i <= n:
            if n % i == 0:
                divs.append(i)
                if i * i != n:
                    divs.append(n // i)
            i += 1
        divs.sort()
        return divs
    
    def score(n):
        if n < 2:
            return 0
        divs = get_divisors(n)
        count = 0
        for i in range(len(divs) - 1):
            if math.gcd(divs[i], divs[i+1]) == 1:
                count += 1
        return count
    
    # 3. Search strategy
    # The space of n <= 10^18 is huge, but we can restrict to numbers with many distinct prime factors.
    # We know 15 primes is the max count. 
    # Let's evaluate the product of the first 15 primes.
    best_n = 1
    best_score = 0
    
    # Candidate 1: Product of first 15 primes
    p15 = 1
    for p in primes[:15]:
        p15 *= p
    if p15 <= 10**18:
        s = score(p15)
        if s > best_score:
            best_score = s
            best_n = p15
    
    # Candidate 2: Maybe replacing a small prime with a larger one allows for a higher power of 2?
    # Or maybe using fewer primes but higher exponents?
    # Let's do a local search around the best candidate.
    # Represent n by its prime factorization: {p: e}
    
    def factorize(n):
        factors = {}
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                e = 0
                while temp % p == 0:
                    temp //= p
                    e += 1
                factors[p] = e
        if temp > 1:
            factors[temp] = 1
        return factors
    
    def reconstruct(factors):
        n = 1
        for p, e in factors.items():
            n *= p ** e
        return n
    
    # Start with the factorization of the best candidate
    current_factors = factorize(best_n)
    
    # Local search: try changing exponents of primes in the factorization
    # Also try swapping a prime for another prime of similar size?
    # Since we want to maximize coprime adjacencies, more distinct primes is generally better.
    # But let's verify if non-squarefree helps.
    
    # Generate neighbors by increasing/decreasing exponents
    # and swapping primes.
    
    visited = set()
    visited.add(tuple(sorted(current_factors.items())))
    
    # Precompute prime list for swapping
    # We can only swap p for q if n * q // p <= 10**18
    
    steps = 0
    max_steps = 500000 # Limit steps to ensure we finish in time
    
    while steps < max_steps:
        if time.time() > deadline - 10:
            break
        
        # Generate neighbors
        neighbors = []
        
        # 1. Change exponents
        for p in list(current_factors.keys()):
            e = current_factors[p]
            # Increase exponent
            if p ** (e + 1) * (best_n // p ** e) <= 10**18:
                new_factors = dict(current_factors)
                new_factors[p] = e + 1
                neighbors.append(new_factors)
            # Decrease exponent
            if e > 1:
                new_factors = dict(current_factors)
                new_factors[p] = e - 1
                neighbors.append(new_factors)
            elif e == 1:
                new_factors = dict(current_factors)
                del new_factors[p]
                if reconstruct(new_factors) >= 2:
                    neighbors.append(new_factors)
            
            # Swap prime p with another prime q not in current_factors
            # q should be of similar magnitude to p to keep n roughly same size
            # We can just iterate through primes not in factors
            base_val = best_n // p
            for q in primes:
                if q in current_factors:
                    continue
                if base_val * q <= 10**18:
                    new_factors = dict(current_factors)
                    del new_factors[p]
                    new_factors[q] = 1
                    neighbors.append(new_factors)
                # Also allow q to have higher exponent? No, start with 1.
                # We can also add a new prime factor
                if base_val * q <= 10**18:
                    new_factors2 = dict(current_factors)
                    new_factors2[q] = 1
                    neighbors.append(new_factors2)
        
        # 2. Add a new prime factor
        for q in primes:
            if q not in current_factors:
                if best_n * q <= 10**18:
                    new_factors = dict(current_factors)
                    new_factors[q] = 1
                    neighbors.append(new_factors)
        
        if not neighbors:
            break
            
        best_neighbor = None
        best_neighbor_score = best_score
        
        for nf in neighbors:
            key = tuple(sorted(nf.items()))
            if key in visited:
                continue
            visited.add(key)
            
            n = reconstruct(nf)
            if n > 10**18 or n < 2:
                continue
            
            s = score(n)
            if s > best_neighbor_score:
                best_neighbor_score = s
                best_neighbor = nf
                best_neighbor_n = n
                best_score_global = s # Keep track of global best
        
        if best_neighbor:
            current_factors = best_neighbor
            best_n = best_neighbor_n
            best_score = best_neighbor_score
            # Optional: Reset visited or keep it? Keep it to avoid cycles.
        else:
            break
            
        steps += 1

    return best_n
# EVOLVE-BLOCK-END
