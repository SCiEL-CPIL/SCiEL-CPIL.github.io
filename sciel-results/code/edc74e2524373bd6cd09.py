"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from itertools import product

def _primes_up_to(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def _count_coprime_adjacencies(n):
    """Count adjacent coprime pairs in sorted divisor list of n."""
    if n < 2:
        return 0
    # Factorize n
    factors = {}
    temp = n
    p = 2
    while p * p <= temp:
        while temp % p == 0:
            factors[p] = factors.get(p, 0) + 1
            temp //= p
        p += 1 if p == 2 else 2
    if temp > 1:
        factors[temp] = factors.get(temp, 0) + 1
    
    # Generate all divisors
    primes = list(factors.keys())
    exponents = [factors[p] for p in primes]
    
    divisors = [1]
    for p, exp in zip(primes, exponents):
        current = []
        for d in divisors:
            for e in range(exp + 1):
                current.append(d * (p ** e))
        divisors = current
    
    divisors.sort()
    
    count = 0
    for i in range(len(divisors) - 1):
        if math.gcd(divisors[i], divisors[i+1]) == 1:
            count += 1
    return count

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # First, try the baseline
    best_n = 2 * 3 * 5 * 7 * 11 * 13 * 17 * 19
    best_score = _count_coprime_adjacencies(best_n)
    
    # Try squarefree products of first k primes for various k
    primes_small = _primes_up_to(200)
    
    # Try different sets of primes
    # Strategy: try all subsets of the first ~20 primes, with product <= 10^18
    # But that's too many. Instead, try structured candidates.
    
    candidates = []
    
    # 1. Squarefree: product of first k primes
    prod = 1
    for i, p in enumerate(primes_small):
        if prod * p > 10**18:
            break
        prod *= p
        candidates.append(prod)
    
    # 2. Try replacing some larger primes with smaller ones + exponents
    # For each k, try n = p1^a1 * p2^a2 * ... where product <= 10^18
    # Focus on small primes with exponents 1 or 2
    
    # Try: for the first m primes, try exponent patterns
    # m up to about 12 (since 2^12 * 3^12 * ... gets big fast)
    
    # Let's try a more targeted search
    # For each number of distinct primes k (2 to 15), try combinations
    
    # Precompute: for squarefree, the score depends on which primes are used
    # Let's evaluate all squarefree n with k primes from the first 20 primes
    
    # Generate all squarefree products of subsets of first 20 primes
    # 2^20 = 1M subsets, but we can prune by product > 10^18
    
    first_20 = primes_small[:20]
    
    # DFS to generate squarefree products <= 10^18
    def gen_squarefree(idx, current_prod, primes_used):
        if time.time() > deadline - 5.0:
            return
        if current_prod > 10**18:
            return
        if idx == len(first_20):
            if current_prod >= 2:
                candidates.append(current_prod)
            return
        # Don't use prime idx
        gen_squarefree(idx + 1, current_prod, primes_used)
        # Use prime idx
        if current_prod * first_20[idx] <= 10**18:
            gen_squarefree(idx + 1, current_prod * first_20[idx], primes_used + [first_20[idx]])
    
    # This could be slow, so let's be more targeted
    # Instead, just try the first k primes for k=2..15, and some variants
    
    # Evaluate the squarefree candidates we already have
    for c in candidates:
        if time.time() > deadline - 5.0:
            break
        s = _count_coprime_adjacencies(c)
        if s > best_score:
            best_score = s
            best_n = c
    
    # Now try mixed exponents: for the first few primes, try exponent 2
    # n = 2^a * 3^b * 5^c * ... with product <= 10^18
    
    # Try small exponent patterns
    # For primes [2,3,5,7,11,13,17,19,23,29,31,37], try exponents in {1,2}
    # But 2^2 * 3^2 * 5^2 * ... * 37^2 is way too big
    # So limit to first ~8 primes with exponents 1 or 2
    
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19]
    
    # Try all combinations of exponents 1 or 2 for these 8 primes
    # 2^8 = 256 combinations
    for exps in product([1, 2], repeat=len(small_primes)):
        if time.time() > deadline - 5.0:
            break
        n = 1
        valid = True
        for p, e in zip(small_primes, exps):
            n *= p ** e
            if n > 10**18:
                valid = False
                break
        if not valid or n < 2:
            continue
        s = _count_coprime_adjacencies(n)
        if s > best_score:
            best_score = s
            best_n = n
    
    # Try adding one more prime with exponent 1
    for exps in product([1, 2], repeat=len(small_primes)):
        if time.time() > deadline - 5.0:
            break
        n = 1
        valid = True
        for p, e in zip(small_primes, exps):
            n *= p ** e
            if n > 10**18:
                valid = False
                break
        if not valid:
            continue
        # Try adding 23, 29, 31, 37, 41, 43, 47 with exponent 1
        for extra_p in [23, 29, 31, 37, 41, 43, 47]:
            if n * extra_p > 10**18:
                continue
            n2 = n * extra_p
            s = _count_coprime_adjacencies(n2)
            if s > best_score:
                best_score = s
                best_n = n2
    
    # Try a few more structured candidates
    # n = product of first k primes for k up to 15 (already in candidates)
    
    return best_n
# EVOLVE-BLOCK-END
