"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START

import numpy as np

DEADLINE = 85.0
_start = time.time()

def _elapsed():
    return time.time() - _start

def _sieve_spf(max_n):
    """Build smallest prime factor array up to max_n."""
    spf = np.arange(max_n + 1, dtype=np.int64)
    spf[0] = 0
    spf[1] = 1
    for i in range(2, int(math.isqrt(max_n)) + 1):
        if spf[i] == i:
            spf[i*i::i] = np.where(spf[i*i::i] == np.arange(i*i, max_n+1, i), i, spf[i*i::i])
    return spf

def _powerful_part_spf(n, spf):
    """Compute powerful part of n using SPF array."""
    if n <= 1:
        return 1
    pp = 1
    temp = n
    while temp > 1:
        p = spf[temp]
        e = 0
        while temp % p == 0:
            temp //= p
            e += 1
        if e >= 2:
            pp *= p ** e
    return pp

def _generate_powerful_numbers(max_val):
    """Generate all powerful numbers (products of distinct prime powers p^e, e>=2) up to max_val."""
    limit = int(math.isqrt(max_val)) + 2
    if limit < 2:
        limit = 2
    # Simple sieve for primes
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    primes = [i for i in range(2, limit + 1) if sieve[i]]
    
    # For each prime, list of powers p^e for e>=2 that are <= max_val
    prime_powers = {}
    for p in primes:
        powers = []
        val = p * p
        e = 2
        while val <= max_val and e <= 12:
            powers.append(val)
            e += 1
            val *= p
        prime_powers[p] = powers
    
    # Generate all products of distinct prime powers
    active_primes = [p for p in primes if p * p <= max_val]
    
    current = [1]
    for p in active_primes:
        if _elapsed() > DEADLINE:
            break
        powers = prime_powers.get(p, [])
        if not powers:
            continue
        new_current = list(current)
        for pw in powers:
            for v in current:
                prod = v * pw
                if prod <= max_val:
                    new_current.append(prod)
        current = new_current
    
    powerful_list = sorted(set(current))
    return powerful_list

def _solve_crt_zero_minus1(A, B):
    """Solve x ≡ 0 (mod A), x ≡ -1 (mod B). Returns (x0, lcm) or None."""
    g = math.gcd(A, B)
    if (B - 1) % g != 0:
        return None
    a1 = A // g
    b1 = B // g
    c1 = (B - 1) // g
    # a1 * k ≡ c1 (mod b1), gcd(a1,b1)=1
    inv_a1 = pow(a1, -1, b1)
    k0 = (c1 * inv_a1) % b1
    x0 = A * k0
    lcm = A * b1
    return x0, lcm

def construct():
    global _start, DEADLINE
    _start = time.time()
    DEADLINE = 85.0
    
    best_score = 0
    best_n = 10000
    
    max_n = 1000000
    
    # Build SPF for fast powerful part computation
    spf = _sieve_spf(max_n + 2)
    
    if _elapsed() > DEADLINE:
        return 10000
    
    # Generate powerful numbers up to max_n
    powerful_nums = _generate_powerful_numbers(max_n)
    
    if _elapsed() > DEADLINE:
        return 10000
    
    # Filter powerful numbers > 1
    powerful_nums = [x for x in powerful_nums if x > 1]
    
    # Sort descending by value
    powerful_nums.sort(reverse=True)
    
    # We want to find pairs (A, B) with A*B maximized, A|n, B|(n+1), gcd(A,B)=1
    # Score is actually powerful_part(n)*powerful_part(n+1) which may exceed A*B
    # if n has additional powerful factors beyond A.
    
    # Strategy: try top powerful numbers as A and B
    # Limit to manageable number of candidates
    K = min(len(powerful_nums), 300)
    candidates = powerful_nums[:K]
    
    # Also add some mid-range for diversity
    if len(powerful_nums) > K:
        mid_start = K
        mid_end = min(len(powerful_nums), K + 100)
        candidates.extend(powerful_nums[mid_start:mid_end])
    
    # Try pairs
    n_candidates_found = 0
    
    for i, A in enumerate(candidates):
        if _elapsed() > DEADLINE:
            break
        for j, B in enumerate(candidates):
            if _elapsed() > DEADLINE:
                break
            if i == j:
                continue
            if math.gcd(A, B) != 1:
                continue
            
            sol = _solve_crt_zero_minus1(A, B)
            if sol is None:
                continue
            x0, lcm = sol
            
            # Find n in [10000, 1000000] with n ≡ x0 (mod lcm)
            if x0 < 10000:
                t = (10000 - x0 + lcm - 1) // lcm
                n_val = x0 + t * lcm
            else:
                n_val = x0
            
            if n_val < 10000 or n_val > max_n:
                continue
            
            # Compute actual powerful parts
            pp_n = _powerful_part_spf(n_val, spf)
            pp_np1 = _powerful_part_spf(n_val + 1, spf)
            
            score = pp_n * pp_np1
            if score > best_score:
                best_score = score
                best_n = n_val
                n_candidates_found += 1
            
            if _elapsed() > DEADLINE:
                return best_n
    
    if best_n != 10000 or best_score > 0:
        return best_n
    
    # Fallback: scan a range
    for n in range(10000, min(100000, max_n + 1)):
        if _elapsed() > DEADLINE:
            break
        pp_n = _powerful_part_spf(n, spf)
        pp_np1 = _powerful_part_spf(n + 1, spf)
        score = pp_n * pp_np1
        if score > best_score:
            best_score = score
            best_n = n
    
    return best_n

# EVOLVE-BLOCK-END
