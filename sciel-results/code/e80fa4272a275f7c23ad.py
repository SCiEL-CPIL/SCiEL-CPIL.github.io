# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    deadline = time.time() + 115.0  # safety margin before 120s limit
    
    # Deterministic Miller-Rabin for 64-bit integers
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        # Check for small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        # Miller-Rabin
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        # Bases sufficient for n < 2^64
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True
    
    # Pollard's Rho with Brent's cycle detection
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(2, n - 1)
            c = random.randint(1, n - 1)
            m = 128
            g, r, q = 1, 1, 1
            while g == 1:
                x = y
                for _ in range(r):
                    y = (y * y + c) % n
                k = 0
                while k < r and g == 1:
                    ys = y
                    for _ in range(min(m, r - k)):
                        y = (y * y + c) % n
                        q = q * abs(x - y) % n
                    g = math.gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                while True:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
                    if g > 1:
                        break
            if g != n:
                return g

    # Check if all prime factors of v are <= limit, with early termination
    def all_factors_leq(v, limit):
        if v <= 1:
            return True
        # Quick check: if v is prime and > limit, fail
        if is_prime(v):
            return v <= limit
        n = v
        # Try small primes first
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for p in small_primes:
            while n % p == 0:
                if p > limit:
                    return False
                n //= p
        if n == 1:
            return True
        if is_prime(n):
            return n <= limit
        # Use Pollard's Rho for remaining composite
        stack = [n]
        while stack:
            m = stack.pop()
            if m == 1:
                continue
            if is_prime(m):
                if m > limit:
                    return False
            else:
                d = pollard_brent(m)
                if d == m:
                    # Retry
                    stack.append(m)
                else:
                    stack.append(d)
                    stack.append(m // d)
        return True

    best_k = 0
    best_p = 2
    
    # Sieve primes up to a limit to iterate efficiently
    def sieve(limit):
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                step = i
                start = i * i
                is_p[start:limit+1:step] = bytearray(len(range(start, limit+1, step)))
        return [i for i in range(2, limit+1) if is_p[i]]

    # Precompute primes for sieve and small factor removal
    # We need primes up to max(p, sqrt(p^2+200)) ~ p
    # But we don't know p yet.
    # Let's generate primes up to 10^6 first.
    primes_small = sieve(10**6)
    
    # Check p = 2
    if time.time() < deadline:
        k = 0
        while time.time() < deadline:
            v = 4 + k + 1
            if all_factors_leq(v, 2):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = 2

    # Batch check function using segmented sieve logic
    # Returns the length of the run of p-smooth numbers starting at p^2 + 1
    def get_smooth_run_length(p, max_k=200):
        if time.time() >= deadline:
            return 0
            
        # We want to check p^2 + 1, ..., p^2 + max_k
        # Use a small set of primes to strip factors.
        # If p is small, we can use all primes <= p.
        # If p is large, we use primes <= 10^5 (or similar) and then check remainders.
        
        # Determine which primes to use for stripping
        # If p <= 10^6, we have all primes <= p.
        # Otherwise, we use primes <= 10^6.
        
        # For efficiency, let's just use primes up to min(p, 10^5) for the bulk stripping.
        # Then for each number, if remainder > 1, we check if it's p-smooth.
        
        # Actually, a simpler approach:
        # For each i in 1..max_k, check if p^2+i is p-smooth.
        # But we can speed up by checking small factors for all i at once?
        # Let's stick to individual checks but optimized with the precomputed small primes list.
        
        k = 0
        for i in range(1, max_k + 1):
            v = p * p + i
            if all_factors_leq(v, p):
                k += 1
            else:
                break
        return k

    # Search primes p > 2
    # We iterate through primes.
    # To maximize k, we should search primes that are likely to have long runs.
    # Heuristic: p should be large enough to cover small factors, but not so large that p^2+i has large prime factors.
    # Actually, the condition is that p^2+i has NO prime factor > p.
    # This is equivalent to saying p^2+i is p-smooth.
    
    # Let's try to search primes in increasing order.
    # The current code does this.
    
    # Optimization: If p is small, p^2 is small, so factorization is fast.
    # If p is large, p^2 is large.
    
    # Let's try to find a good p.
    # The current code iterates all primes up to 10^6, then 10^7.
    # This is exhaustive but might be slow if many primes are checked.
    # Each check involves up to 200 factorizations.
    # If most fail at i=1 or i=2, it's fast.
    
    # Let's add a quick filter:
    # For p^2+1 to be p-smooth, it must not have any prime factor > p.
    # If p^2+1 is prime, it must be <= p. But p^2+1 > p for p>1.
    # So p^2+1 must be composite.
    # If p^2+1 has a prime factor q > p, it fails.
    
    # We can pre-check if p^2+1 has any small prime factor.
    # If it does, we can divide it out and check the rest.
    
    # The current implementation does this in all_factors_leq.
    
    # Let's try to improve the search order or add a heuristic.
    # Maybe search larger primes first? Or random sampling?
    # The task says "Maximize k".
    # Long runs are rare.
    
    # Let's stick to the current search but optimize the inner loop slightly.
    # The main cost is in all_factors_leq.
    
    # One improvement: In all_factors_leq, we can check divisibility by primes up to min(limit, sqrt(v)) using the precomputed list if possible.
    # But all_factors_leq uses a hardcoded small_primes list.
    # Let's modify all_factors_leq to accept a list of primes to try.
    
    def all_factors_leq_optimized(v, limit, primes_to_try):
        if v <= 1:
            return True
        if is_prime(v):
            return v <= limit
        n = v
        for p in primes_to_try:
            if p > limit:
                break
            if p * p > n:
                break
            while n % p == 0:
                n //= p
        if n == 1:
            return True
        if is_prime(n):
            return n <= limit
        # If n is composite, it has a factor > max(primes_to_try)
        # If max(primes_to_try) >= sqrt(n), then n is prime (handled above).
        # So if we get here, n is composite and has a factor > max(primes_to_try).
        # We need to check if that factor is <= limit.
        # If limit >= max(primes_to_try), we might have missed factors.
        # In our usage, primes_to_try will be primes <= min(limit, 10^5).
        # If limit > 10^5, we might miss factors in (10^5, limit].
        # So we must be careful.
        
        # For safety, if limit > 10^5, we fall back to Pollard.
        # But if limit <= 10^5, and we tried all primes <= limit, then n must be 1 or prime.
        # So if we get here, it means limit > 10^5 (or primes_to_try didn't cover up to limit).
        
        # Let's just use the original all_factors_leq for safety, but pass the primes list to speed up the initial division.
        # Actually, the hardcoded list in all_factors_leq is small (up to 97).
        # Let's extend it or pass a larger list.
        
        return False # Placeholder, will replace with full logic

    # Let's refine all_factors_leq to use a larger precomputed prime list for trial division.
    # We have primes_small up to 10^6.
    # We can pass a slice of this list.
    
    def all_factors_leq_v2(v, limit):
        if v <= 1:
            return True
        if is_prime(v):
            return v <= limit
        n = v
        # Use primes from primes_small up to min(limit, sqrt(n))
        # But we don't want to iterate 10^6 primes for every call.
        # Let's use primes up to 10^3 or 10^4 for trial division.
        # primes_small[0:1000] approx 1000 primes.
        
        # Find index of 10^4 in primes_small
        # primes_small is sorted.
        # Let's just use the first 1000 primes (up to ~7900).
        # This covers most small factors.
        
        limit_idx = 1000 # ~7900
        for i in range(limit_idx):
            p = primes_small[i]
            if p > limit:
                break
            if p * p > n:
                break
            while n % p == 0:
                n //= p
        if n == 1:
            return True
        if is_prime(n):
            return n <= limit
        # If n is composite, use Pollard
        stack = [n]
        while stack:
            m = stack.pop()
            if m == 1:
                continue
            if is_prime(m):
                if m > limit:
                    return False
            else:
                d = pollard_brent(m)
                if d == m:
                    stack.append(m)
                else:
                    stack.append(d)
                    stack.append(m // d)
        return True

    # Update the loop to use all_factors_leq_v2
    
    for p in primes_small:
        if time.time() >= deadline:
            break
        if p == 2:
            continue
        k = 0
        while time.time() < deadline:
            v = p * p + k + 1
            if all_factors_leq_v2(v, p):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
            # Optional: if best_k is high, we might want to search around this p?
            # But for now, just continue.
            
    # If we still have time, we can search larger primes
    if time.time() < deadline - 10:
        limit2 = 10**7
        primes2 = sieve(limit2)
        start_idx = len(primes_small)
        for p in primes2[start_idx:]:
            if time.time() >= deadline:
                break
            k = 0
            while time.time() < deadline:
                v = p * p + k + 1
                if all_factors_leq_v2(v, p):
                    k += 1
                else:
                    break
            if k > best_k:
                best_k = k
                best_p = p

    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
