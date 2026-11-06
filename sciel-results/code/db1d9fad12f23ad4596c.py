"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import sys

# EVOLVE-BLOCK-START
import numpy as np

_DEADLINE_MARGIN = 5  # seconds of safety margin before the 90s kill

def _factorize(n):
    """Return list of (prime, exponent) for n."""
    if n <= 1:
        return []
    factors = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            e = 0
            while n % d == 0:
                n //= d
                e += 1
            factors.append((d, e))
        d += 1 if d == 2 else 2  # skip evens after 2
    if n > 1:
        factors.append((n, 1))
    return factors

def _powerful_part(n):
    """Product of p^e for all p^e || n with e >= 2."""
    if n <= 1:
        return 1
    pp = 1
    d = 2
    while d * d <= n:
        if n % d == 0:
            e = 0
            while n % d == 0:
                n //= d
                e += 1
            if e >= 2:
                pp *= d ** e
        d += 1 if d == 2 else 2
    if n > 1:
        # remaining factor has exponent 1
        pass
    return pp

def _powerful_part_fast(n, smallest_prime_factor):
    """Use SPF table if available; fall back to trial division."""
    if smallest_prime_factor is not None and n < len(smallest_prime_factor):
        pp = 1
        x = n
        while x > 1:
            p = smallest_prime_factor[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                pp *= p ** e
        return pp
    else:
        return _powerful_part(n)

def _build_spf(limit):
    """Build smallest-prime-factor sieve up to limit."""
    spf = list(range(limit + 1))
    if limit >= 0:
        spf[0] = 0
    if limit >= 1:
        spf[1] = 1
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def _score_candidate(n, spf):
    """Return powerful_part(n) * powerful_part(n+1)."""
    if n < 10000 or n + 1 > 1000000:
        return 0
    ppn = _powerful_part_fast(n, spf)
    ppn1 = _powerful_part_fast(n + 1, spf)
    return ppn * ppn1

def _search_random(n_trials, spf, deadline):
    """Random sampling in [10000, 999999]."""
    best_n = 10000
    best_score = _score_candidate(10000, spf)
    for _ in range(n_trials):
        if time.time() > deadline:
            break
        n = random.randint(10000, 999999)
        s = _score_candidate(n, spf)
        if s > best_score:
            best_score = s
            best_n = n
    return best_n, best_score

def _search_local_improve(n0, spf, deadline, max_steps=200000):
    """Hill-climb from n0 by +-1, +-2, ... small jumps."""
    best_n = n0
    best_score = _score_candidate(n0, spf)
    step = 0
    while step < max_steps and time.time() < deadline:
        improved = False
        for delta in (1, -1, 2, -2, 3, -3, 5, -5, 10, -10, 100, -100, 1000, -1000):
            m = best_n + delta
            if m < 10000 or m + 1 > 1000000:
                continue
            s = _score_candidate(m, spf)
            if s > best_score:
                best_score = s
                best_n = m
                improved = True
                break
        if not improved:
            # random jump to escape local optimum
            for _ in range(10):
                if time.time() > deadline:
                    break
                m = random.randint(10000, 999999)
                s = _score_candidate(m, spf)
                if s > best_score:
                    best_score = s
                    best_n = m
                    improved = True
                    break
        if not improved:
            break
        step += 1
    return best_n, best_score

def _search_structural(spq, deadline):
    """
    Structural search: pick primes p, q and small exponents a, b >= 2,
    construct n = p^a * s where s is squarefree-ish, and n+1 = q^b * t.
    We look for consecutive pairs where both have a large powerful part.
    Strategy: enumerate p^a * q^b forms for n and n+1 separately? That's hard.
    Instead: for each small prime p, try n = k * p^e with e>=2, check powerful part of n and n+1.
    """
    best_n = 10000
    best_score = 0
    
    # Precompute some useful prime powers
    prime_powers = {}  # pp_val -> (prime, exp)
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in primes:
        pk = p * p
        while pk <= 1000000:
            prime_powers[pk] = (p, 2)
            pk *= p
            if (p, 3) not in [(pp[0], pp[1]) for pp in prime_powers.values()]:
                pass
    
    # For each prime power P = p^e (e>=2), look for n in [10000, 999999] such that n % P == 0
    # and the remaining factor of n contributes minimally to powerful part (ideally squarefree),
    # and n+1 also has a good powerful part.
    # We'll iterate over P and check a few multiples.
    
    for P, (p, e) in sorted(prime_powers.items()):
        if time.time() > deadline:
            break
        if P < 100:  # skip tiny prime powers, unlikely to dominate
            continue
        # n must be a multiple of P: n = P * k, 10000 <= P*k <= 999999
        k_min = max(1, (10000 + P - 1) // P)
        k_max = 999999 // P
        if k_min > k_max:
            continue
        # Check a sample of k values
        n_samples = []
        if k_max - k_min <= 50:
            n_samples = list(range(k_min, k_max + 1))
        else:
            n_samples = list(range(k_min, k_max + 1, max(1, (k_max - k_min) // 50)))
            # Always include k_min and k_max
            if k_max not in n_samples:
                n_samples.append(k_max)
        
        for k in n_samples:
            if time.time() > deadline:
                break
            n = P * k
            if n < 10000 or n + 1 > 1000000:
                continue
            # Quick check: is k squarefree? If so, powerful_part(n) = P (assuming no other squares in k)
            # Actually powerful_part(n) includes P and any p'^f from k with f>=2.
            # For efficiency, just compute it.
            s = _score_candidate(n, None)
            if s > best_score:
                best_score = s
                best_n = n
    return best_n, best_score

def _search_consecutive_powerful_pairs(deadline):
    """
    Look for n such that n and n+1 both have large powerful parts.
    Use the property: if n = a^2 * b^3 ... and n+1 = c^2 * d^3 ...,
    try known forms like n = x^2, n+1 = y^2 is impossible for x,y>1.
    But n = x^2 * 2, n+1 = y^2 * 3, etc.
    Brute force over squares and cubes in range.
    """
    best_n = 10000
    best_score = 0
    
    # Generate all numbers in [10000, 1000000] that are divisible by a high prime power
    # and check their neighbors.
    # We already do this in structural search. Let's do a denser scan for medium P.
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31]
    for p in primes:
        e = 2
        while True:
            P = p ** e
            if P > 1000000:
                break
            # For each multiple n = P*k in range, check score
            k_min = max(1, (10000 + P - 1) // P)
            k_max = 999999 // P
            if k_min > k_max:
                e += 1
                continue
            # Sample
            step = max(1, (k_max - k_min + 1) // 200)
            for k in range(k_min, k_max + 1, step):
                if time.time() > deadline:
                    return best_n, best_score
                n = P * k
                s = _score_candidate(n, None)
                if s > best_score:
                    best_score = s
                    best_n = n
            e += 1
    return best_n, best_score

def construct():
    start_time = time.time()
    total_deadline = start_time + 85  # 5s margin
    
    # Strategy: Exhaustive scan using precomputed powerful parts array.
    # This guarantees finding the global maximum.
    
    N_MAX = 1000000
    N_MIN = 10000
    
    # 1. Build SPF up to N_MAX
    spf = _build_spf(N_MAX)
    
    # 2. Compute powerful part for all i in [1, N_MAX]
    # PP[i] = product of p^e for p^e || i with e >= 2
    PP = np.ones(N_MAX + 1, dtype=np.int64)
    
    # Efficient calculation:
    # For each prime p, for each power p^k (k>=2), multiply PP[multiple] by p^k
    # But we must be careful: if i has p^3, we want to multiply by p^3, not p^2 * p^2.
    # The definition is: product of FULL prime powers.
    # So if i = p^3 * q, powerful_part(i) includes p^3.
    # If i = p^4, powerful_part(i) includes p^4.
    # If i = p^2 * q^2, powerful_part(i) includes p^2 * q^2.
    
    # Let's use the SPF to factor each number and compute PP[i].
    # This is O(N log N) roughly, which is fast enough for N=1e6.
    
    # Optimization: We can compute PP[i] iteratively or via sieve-like updates.
    # However, direct factorization using SPF is simple and fast enough.
    
    # Let's try a sieve-like approach for PP.
    # Initialize PP[i] = 1.
    # For each prime p:
    #   For each power p^k (k>=2):
    #     For each multiple m of p^k:
    #       We need to know the exact exponent of p in m.
    #       If exponent is exactly k, multiply PP[m] by p^k.
    #       If exponent > k, we shouldn't multiply by p^k now, but by p^(exact_exp) later?
    #       Actually, it's easier to just factor each number once.
    
    # Let's factor each number 1..N_MAX using SPF.
    # This will take a few seconds in Python but is safe.
    
    # To speed up, we can use a helper that computes PP for all numbers.
    # Since N=1e6, a pure Python loop might take ~5-10s.
    # We have 85s budget. This is safe.
    
    # Let's implement a fast batch computation.
    
    # Method:
    # For i from 1 to N_MAX:
    #   x = i
    #   pp = 1
    #   while x > 1:
    #     p = spf[x]
    #     e = 0
    #     while x % p == 0:
    #       x //= p
    #       e += 1
    #     if e >= 2:
    #       pp *= p ** e
    #   PP[i] = pp
    
    # This is O(N * number_of_prime_factors). Average is small.
    
    # Let's do it.
    
    # We can potentially vectorize the "while x > 1" part? No, it's dependent.
    # But 1e6 iterations in Python is ~0.1-0.5s if optimized? 
    # Actually, 1e6 * ~5 ops = 5e6 ops. Python does ~10e6-50e6 simple ops/sec.
    # So ~0.1-0.5s. Very safe.
    
    for i in range(1, N_MAX + 1):
        x = i
        pp = 1
        while x > 1:
            p = spf[x]
            e = 0
            while x % p == 0:
                x //= p
                e += 1
            if e >= 2:
                pp *= p ** e
        PP[i] = pp
        
    # 3. Compute scores for all n in [N_MIN, N_MAX-1]
    # score[n] = PP[n] * PP[n+1]
    
    # We can do this with numpy slicing
    # scores = PP[N_MIN:N_MAX] * PP[N_MIN+1:N_MAX+1]
    # This creates an array of size ~990000.
    
    scores = PP[N_MIN:N_MAX] * PP[N_MIN+1:N_MAX+1]
    
    # 4. Find the index of the maximum score
    max_idx = np.argmax(scores)
    best_n = N_MIN + max_idx
    best_score = scores[max_idx]
    
    # 5. Return best_n
    # Sanity check
    if best_n < N_MIN or best_n + 1 > N_MAX:
        return N_MIN
        
    return best_n

# EVOLVE-BLOCK-END
