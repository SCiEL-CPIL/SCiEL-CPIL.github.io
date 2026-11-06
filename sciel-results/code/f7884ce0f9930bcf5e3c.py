"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START

import numpy as np

DEADLINE = 85.0
_start = time.time()

def _elapsed():
    return time.time() - _start

def _sieve_primes(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def _generate_powerful_numbers(max_val, primes=None):
    """
    Generate powerful numbers up to max_val.
    A powerful number is a product of prime powers p^e with e>=2,
    where each prime appears at most once in the product.
    """
    if primes is None:
        limit = int(math.isqrt(max_val)) + 2
        primes = _sieve_primes(limit)
    
    # For each prime, list of powers p^e for e>=2 that are <= max_val
    prime_powers = {}
    for p in primes:
        if p * p > max_val:
            break
        powers = []
        val = p * p
        e = 2
        while val <= max_val and e <= 12:
            powers.append(val)
            e += 1
            val *= p
        if powers:
            prime_powers[p] = powers
    
    # Generate all products of distinct prime powers
    current = [1]
    for p in prime_powers:
        powers = prime_powers[p]
        new_current = list(current)  # keep the version not using this prime
        for pw in powers:
            for v in current:
                prod = v * pw
                if prod <= max_val:
                    new_current.append(prod)
        current = new_current
    
    powerful_list = sorted(set(current))
    return powerful_list

def _powerful_part(n, small_primes=None):
    """Compute the powerful part of n: product of p^e for each prime p dividing n with exponent e>=2."""
    if n <= 1:
        return 1
    temp = n
    pp = 1
    d = 2
    while d * d <= temp:
        if temp % d == 0:
            e = 0
            while temp % d == 0:
                temp //= d
                e += 1
            if e >= 2:
                pp *= d ** e
        d += 1
    if temp > 1:
        # temp is prime with exponent 1, so not powerful
        pass
    return pp

def _solve_crt(a, b):
    """Solve x ≡ 0 mod a, x ≡ -1 mod b. Returns (x0, lcm) or None if no solution."""
    g = math.gcd(a, b)
    if (b - 1) % g != 0:
        return None
    a1 = a // g
    b1 = b // g
    c1 = (b - 1) // g
    inv_a1 = pow(a1, -1, b1)
    k0 = (c1 * inv_a1) % b1
    x0 = a * k0
    lcm = a * b1
    return x0, lcm

def construct():
    global _start, _elapsed, DEADLINE
    _start = time.time()
    DEADLINE = 85.0
    
    best_score = 0
    best_n = 10000
    
    # Generate powerful numbers up to 1000000
    max_mod = 1000000
    primes = _sieve_primes(int(math.isqrt(max_mod)) + 2)
    powerful_nums = _generate_powerful_numbers(max_mod, primes)
    
    if _elapsed() > DEADLINE:
        return 10000
    
    # Filter: we want powerful numbers > 1
    powerful_nums = [x for x in powerful_nums if x > 1]
    powerful_nums.sort()
    
    # Strategy: For coprime pairs (A, B) with A*B <= 1000000 (since lcm=A*B for coprime),
    # solve CRT and check if solution is in [10000, 1000000].
    # Score lower bound is A*B.
    
    # To be efficient, we'll iterate over pairs where A*B <= 1000000
    # Sort powerful numbers for binary search
    pn = powerful_nums
    
    # Try pairs in order of decreasing A*B
    # But we need to be smart about which pairs to try
    
    # First pass: try all pairs where A and B are both "large"
    # A powerful number A means n is divisible by A, so A <= n <= 1000000
    # Similarly B <= n+1 <= 1000001
    
    # For coprime A, B: lcm = A*B, and we need A*B <= ~1000000 for a solution to exist
    # (actually A*B can be slightly larger but let's keep it simple)
    
    # Group powerful numbers by size for efficient pairing
    # Try to maximize A*B subject to A*B <= 1000000 and gcd(A,B)=1
    
    # Sort in descending order
    pn_desc = sorted(pn, reverse=True)
    
    # Limit candidates to keep search tractable
    # Focus on powerful numbers that are "useful" - not too small
    # A powerful number of size ~1000 paired with ~1000 gives product ~1e6
    # So we want numbers in range [10, 1000000]
    
    # Let's try a multi-pass approach:
    # Pass 1: Try pairs where both A and B are > 100
    # Pass 2: Try pairs where one is > 100 and other is > 10
    # etc.
    
    # For efficiency, limit the number of candidates we consider
    # The number of powerful numbers up to 1e6 is manageable (~few thousand)
    
    n_pn = len(pn_desc)
    
    # Try pairs, prioritizing larger products
    # Use a two-pointer or sorted approach
    
    # Actually, let's just iterate over all pairs where A*B <= 1000000
    # and gcd(A,B) = 1, checking in order of decreasing A*B
    
    # Build list of (A, B, product) for valid coprime pairs
    # This could be large, so let's be selective
    
    # Better approach: for each A (from large to small), find compatible B
    # B must be coprime to A and B <= 1000000//A
    
    # Precompute for each A, the max B we can pair with
    max_total = 1000000
    
    # Iterate A from largest to smallest
    # For each A, B_max = max_total // A
    # Find powerful numbers B <= B_max that are coprime to A
    
    # To make this faster, precompute gcd relationships
    # But that's expensive for large sets. Instead, just check gcd on the fly.
    
    # Limit the number of A's we try (largest first)
    max_A_tries = min(n_pn, 500)
    
    for i in range(max_A_tries):
        if _elapsed() > DEADLINE:
            break
        A = pn_desc[i]
        B_max = max_total // A
        if B_max < 2:
            continue
        
        # Find powerful numbers B <= B_max, B != A, gcd(A,B)=1
        # pn is sorted ascending, so use binary search
        # Find rightmost index where pn[j] <= B_max
        lo, hi = 0, n_pn - 1
        right_idx = -1
        while lo <= hi:
            mid = (lo + hi) // 2
            if pn[mid] <= B_max:
                right_idx = mid
                lo = mid + 1
            else:
                hi = mid - 1
        
        if right_idx < 0:
            continue
        
        # Try B values from pn[0..right_idx]
        # Prioritize larger B (so iterate from right_idx down)
        # But limit the number of B's per A
        max_B_per_A = 50
        
        for j in range(right_idx, -1, -1):
            if _elapsed() > DEADLINE:
                break
            B = pn[j]
            if B == A:
                continue
            if B > B_max:
                continue
            
            # Check coprime
            if math.gcd(A, B) != 1:
                continue
            
            # Solve CRT: x ≡ 0 mod A, x ≡ -1 mod B
            sol = _solve_crt(A, B)
            if sol is None:
                continue
            x0, lcm = sol
            
            # Find n in [10000, 1000000] such that n ≡ x0 (mod lcm)
            # n = x0 + t*lcm
            # 10000 <= x0 + t*lcm <= 1000000
            # t_min = ceil((10000 - x0) / lcm)
            # t_max = floor((1000000 - x0) / lcm)
            
            if lcm <= 0:
                continue
            
            t_min = 0
            if x0 < 10000:
                t_min = (10000 - x0 + lcm - 1) // lcm
            t_max = (1000000 - x0) // lcm
            
            if t_max < t_min:
                continue
            
            # Pick the smallest valid n (or any valid n)
            n_val = x0 + t_min * lcm
            
            # Verify actual powerful parts
            pp_n = _powerful_part(n_val)
            pp_np1 = _powerful_part(n_val + 1)
            
            score = pp_n * pp_np1
            if score > best_score:
                best_score = score
                best_n = n_val
            
            # Try a few more t values for this (A,B) pair
            for t in range(t_min + 1, min(t_min + 5, t_max + 1)):
                if _elapsed() > DEADLINE:
                    break
                n_val2 = x0 + t * lcm
                pp_n2 = _powerful_part(n_val2)
                pp_np12 = _powerful_part(n_val2 + 1)
                score2 = pp_n2 * pp_np12
                if score2 > best_score:
                    best_score = score2
                    best_n = n_val2
            
            # Limit B iterations per A
            if right_idx - j >= max_B_per_A:
                break
    
    if best_n != 10000 or best_score > 0:
        return best_n
    
    return 10000

# EVOLVE-BLOCK-END
