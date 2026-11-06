"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
from functools import lru_cache

# EVOLVE-BLOCK-START
import sys

def powerful_part(n):
    """Compute the powerful part of n: product of p^e where e>=2."""
    if n <= 1:
        return 1
    result = 1
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            e = 0
            while temp % p == 0:
                temp //= p
                e += 1
            if e >= 2:
                result *= p ** e
        p += 1 if p == 2 else 2
    if temp > 1:
        # temp is prime with exponent 1, doesn't contribute
        pass
    return result

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    # Precompute small prime powers for efficiency
    # We'll search for n where both n and n+1 have large powerful parts
    
    best_n = 10000
    best_score = powerful_part(10000) * powerful_part(10001)
    
    # Strategy: Look for numbers that are products of prime powers
    # For each combination of prime powers, check if consecutive numbers work
    
    # Generate candidate powerful numbers up to 1000000
    # A powerful number is one where every prime factor has exponent >= 2
    
    # Let's do a direct search but be smart about it
    # Focus on numbers with high powerful parts
    
    # Precompute powerful parts for a range using sieve-like approach
    # But 1000000 is large, so let's be selective
    
    # Better approach: iterate through candidate n values
    # and compute powerful parts efficiently
    
    # We'll scan through the range, computing powerful parts
    # To speed up, we can skip numbers that are clearly not good candidates
    
    # Actually, let's just scan the entire range efficiently
    # powerful_part computation for each number up to 1000000
    
    # Use a modified sieve to compute powerful parts for all numbers in range
    # This is more efficient than factoring each number individually
    
    max_n = 1000000
    min_n = 10000
    
    # Sieve approach for powerful parts
    # For each prime p, for each power p^2, p^3, ... mark multiples
    # and accumulate the powerful part contribution
    
    pp = [1] * (max_n + 2)  # powerful part for each number
    
    # For each prime p, add p^e to the powerful part of multiples of p^e
    # where e >= 2
    
    # First, get primes up to sqrt(max_n) ~ 1000
    limit = int(math.isqrt(max_n)) + 1
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit+1, i):
                is_prime[j] = False
    primes = [i for i in range(2, limit+1) if is_prime[i]]
    
    # For each prime, compute its powers and update multiples
    for p in primes:
        p2 = p * p
        if p2 > max_n:
            break
        # For each multiple of p2, add p2 to its powerful part
        # But we need to handle higher powers too
        # For n divisible by p^e where e>=2, the powerful part includes p^e
        
        # We'll handle this by: for each power p^k (k>=2), 
        # for multiples of p^k that are NOT divisible by p^(k+1),
        # multiply their powerful part by p^k
        
        # Actually simpler: for each n, the powerful part contribution from p
        # is p^(v_p(n)) if v_p(n) >= 2, else 1
        
        # Let's do it differently: for each prime p, for each multiple m of p^2,
        # compute v_p(m) and multiply pp[m] by p^(v_p(m))
        
        # But this is O(n log n) per prime, which might be too slow
        # Let's just do direct computation for a subset of promising candidates
        
    # Given time constraints, let's do a direct scan with early termination
    # Focus on the range and compute powerful parts on the fly
    
    # Optimize: only check n where n or n+1 is divisible by small prime squares
    # This filters out many candidates
    
    # Actually, let's just scan efficiently
    # The key is to find pairs with high product of powerful parts
    
    # Let's scan from 10000 to 1000000, but break early if time is running out
    # and keep track of the best
    
    # To speed up powerful_part, we can memoize or use a faster method
    
    # Let's use a hybrid: precompute for small numbers, compute on fly for large
    
    # Given the 90s limit, let's scan as much as we can
    # We'll scan in chunks and check time periodically
    
    n = min_n
    chunk_size = 10000
    
    while n <= max_n:
        if time.time() > deadline:
            break
        
        # Check a chunk
        for i in range(n, min(n + chunk_size, max_n + 1)):
            pp_i = powerful_part(i)
            pp_i1 = powerful_part(i + 1)
            score = pp_i * pp_i1
            if score > best_score:
                best_score = score
                best_n = i
        
        n += chunk_size
    
    return best_n

# EVOLVE-BLOCK-END
