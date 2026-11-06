"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from collections import defaultdict

# EVOLVE-BLOCK-START

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

def _factorize_with_spf(n, spf):
    """Return dict of prime->exponent for n."""
    factors = {}
    temp = n
    while temp > 1:
        p = spf[temp]
        e = 0
        while temp % p == 0:
            temp //= p
            e += 1
        factors[p] = e
    return factors

def construct():
    global _start, DEADLINE
    _start = time.time()
    DEADLINE = 85.0
    
    best_score = 0
    best_n = 10000
    
    max_n = 1000000
    min_n = 10000
    
    # Build SPF for fast powerful part computation
    spf = _sieve_spf(max_n + 2)
    
    if _elapsed() > DEADLINE:
        return 10000
    
    # Phase 1: Full scan of [10000, 1000000] to find top candidates
    # This is the structural change: instead of only CRT-based search,
    # we do a full scan first, then refine around promising regions.
    
    # We'll scan in chunks to stay within time budget
    # First, do a full scan to find the global best
    scan_start = min_n
    scan_end = max_n + 1
    
    # For efficiency, we can vectorize parts of this, but let's do it in Python
    # with early termination checks
    
    # Actually, let's do a smarter approach:
    # 1. Generate powerful numbers
    # 2. For each powerful number P, find multiples of P in range
    # 3. Check if n+1 also has large powerful part
    
    powerful_nums = _generate_powerful_numbers(max_n)
    
    if _elapsed() > DEADLINE:
        return 10000
    
    # Filter powerful numbers > 1
    powerful_nums = [x for x in powerful_nums if x > 1]
    
    # Sort descending by value
    powerful_nums.sort(reverse=True)
    
    # Phase 2: For each powerful number A (as powerful part of n),
    # find n in range such that A | n and n+1 has large powerful part
    # We'll use the top K powerful numbers as candidates for A
    
    K = min(len(powerful_nums), 500)
    candidates = powerful_nums[:K]
    
    # For each candidate A, find n ≡ 0 (mod A) in [min_n, max_n]
    # and check powerful_part(n+1)
    
    # To speed this up, for each A, the values of n are:
    # n = k*A where k*A in [min_n, max_n]
    # So k in [ceil(min_n/A), floor(max_n/A)]
    
    # For each such n, compute powerful_part(n+1)
    # To make this efficient, we can precompute powerful parts for all n+1
    # but that's 1 million values. Let's just compute on the fly.
    
    # Alternative: For each B in powerful_nums, find n ≡ -1 (mod B)
    # i.e., n = k*B - 1
    # Then check if powerful_part(n) is large.
    
    # Let's try both directions and take the best.
    
    # Direction 1: Fix A (powerful part of n), scan n = k*A
    for A in candidates:
        if _elapsed() > DEADLINE:
            break
        
        k_min = (min_n + A - 1) // A
        k_max = max_n // A
        
        if k_min > k_max:
            continue
        
        # For large A, there are few multiples. For small A, many.
        # Limit the number of multiples we check
        num_multiples = k_max - k_min + 1
        if num_multiples > 10000:
            # Sample or limit
            # For very small A, we can't check all, so skip or sample
            # Actually, for small A, the powerful part of n is at least A,
            # but might be larger. Let's just check a few.
            # Use stride to limit checks
            stride = max(1, num_multiples // 1000)
            k_values = range(k_min, k_max + 1, stride)
        else:
            k_values = range(k_min, k_max + 1)
        
        for k in k_values:
            n_val = k * A
            if n_val < min_n or n_val > max_n:
                continue
            
            # Compute powerful part of n+1
            pp_np1 = _powerful_part_spf(n_val + 1, spf)
            
            # The actual powerful part of n is at least A, but could be more
            # Let's compute it to be accurate
            pp_n = _powerful_part_spf(n_val, spf)
            
            score = pp_n * pp_np1
            if score > best_score:
                best_score = score
                best_n = n_val
    
    if _elapsed() > DEADLINE:
        return best_n
    
    # Direction 2: Fix B (powerful part of n+1), scan n = k*B - 1
    for B in candidates:
        if _elapsed() > DEADLINE:
            break
        
        # n = k*B - 1, so k*B = n+1
        # n+1 in [min_n+1, max_n+1]
        # k*B in [min_n+1, max_n+1]
        k_min = (min_n + 1 + B - 1) // B
        k_max = (max_n + 1) // B
        
        if k_min > k_max:
            continue
        
        num_multiples = k_max - k_min + 1
        if num_multiples > 10000:
            stride = max(1, num_multiples // 1000)
            k_values = range(k_min, k_max + 1, stride)
        else:
            k_values = range(k_min, k_max + 1)
        
        for k in k_values:
            n_val = k * B - 1
            if n_val < min_n or n_val > max_n:
                continue
            
            pp_n = _powerful_part_spf(n_val, spf)
            pp_np1 = _powerful_part_spf(n_val + 1, spf)
            
            score = pp_n * pp_np1
            if score > best_score:
                best_score = score
                best_n = n_val
    
    if _elapsed() > DEADLINE:
        return best_n
    
    # Phase 3: CRT-based refinement
    # For pairs (A, B) with gcd(A,B)=1, find n ≡ 0 (mod A), n ≡ -1 (mod B)
    # This gives n where A | n and B | n+1
    # The powerful part of n is at least A, powerful part of n+1 is at least B
    
    # Use top M powerful numbers for both A and B
    M = min(len(powerful_nums), 200)
    crt_candidates = powerful_nums[:M]
    
    for i, A in enumerate(crt_candidates):
        if _elapsed() > DEADLINE:
            break
        
        for j, B in enumerate(crt_candidates):
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
            
            # Find n in [min_n, max_n] with n ≡ x0 (mod lcm)
            if x0 < min_n:
                t = (min_n - x0 + lcm - 1) // lcm
                n_val = x0 + t * lcm
            else:
                n_val = x0
            
            if n_val < min_n or n_val > max_n:
                continue
            
            pp_n = _powerful_part_spf(n_val, spf)
            pp_np1 = _powerful_part_spf(n_val + 1, spf)
            
            score = pp_n * pp_np1
            if score > best_score:
                best_score = score
                best_n = n_val
    
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
