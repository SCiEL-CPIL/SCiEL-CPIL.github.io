"""Rewritten program: congruence pre-sieve + fast factorization for long p-smooth runs."""
import time
import math
import random
from math import gcd

def construct():
    deadline = time.time() + 135  # safety margin before 150s kill
    MAX_K = 200
    
    # --- Miller-Rabin primality test (deterministic for n < 3.3 * 10^24) ---
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        # Bases sufficient for n < 3.3 * 10^24
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = x * x % n
                if x == n - 1:
                    break
            else:
                return False
        return True
    
    # --- Pollard-Brent factorization ---
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randrange(2, n - 1)
            c = random.randrange(1, n - 1)
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
                    g = gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = gcd(abs(x - ys), n)
            if g != n:
                return g
    
    def factorize(n):
        """Return list of prime factors of n."""
        if n == 1:
            return []
        factors = []
        stack = [n]
        while stack:
            num = stack.pop()
            if num == 1:
                continue
            if is_prime(num):
                factors.append(num)
            else:
                d = pollard_brent(num)
                stack.append(d)
                stack.append(num // d)
        return factors
    
    def max_prime_factor(n):
        """Return the largest prime factor of n."""
        if n == 1:
            return 1
        # Quick trial division for small primes
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                while n % p == 0:
                    n //= p
                if n == 1:
                    return p
        if is_prime(n):
            return n
        # Use Pollard-Brent
        d = pollard_brent(n)
        return max(max_prime_factor(d), max_prime_factor(n // d))
    
    # --- Segmented sieve to generate primes in a range ---
    def primes_in_range(start, end):
        if start < 2:
            start = 2
        if end < start:
            return []
        # Sieve up to sqrt(end)
        limit = int(math.isqrt(end)) + 1
        small_sieve = bytearray(b'\x01') * (limit + 1)
        small_sieve[0] = small_sieve[1] = 0
        for i in range(2, int(math.isqrt(limit)) + 1):
            if small_sieve[i]:
                small_sieve[i*i::i] = bytearray(len(small_sieve[i*i::i]))
        small_primes = [i for i in range(2, limit + 1) if small_sieve[i]]
        
        # Segmented sieve
        size = end - start + 1
        sieve = bytearray(b'\x01') * size
        if start == 1:
            sieve[0] = 0
        for p in small_primes:
            if p * p > end:
                break
            # Find first multiple of p >= start
            first = ((start + p - 1) // p) * p
            if first == p:
                first += p
            for j in range(first - start, size, p):
                sieve[j] = 0
        return [start + i for i in range(size) if sieve[i]]
    
    # --- Congruence pre-sieve ---
    # For a given prime q, find residues r mod q such that p^2 + j ≡ 0 (mod q) for some j in [1, MAX_K]
    # i.e., p^2 ≡ -j (mod q) for some j in [1, MAX_K]
    # We mark p as "bad" if p mod q is in the bad set.
    
    # We'll use small primes q to pre-sieve candidates.
    # For efficiency, we precompute bad residues for a set of small primes q.
    
    def compute_bad_residues(q, max_k=MAX_K):
        """Return set of residues r mod q such that p^2 + j ≡ 0 mod q for some 1<=j<=max_k."""
        bad = set()
        for j in range(1, max_k + 1):
            # Need p^2 ≡ -j (mod q)
            target = (-j) % q
            # Check if target is a quadratic residue mod q
            # For prime q, use Euler's criterion or just try sqrt
            # For small q, just iterate
            for r in range(q):
                if (r * r) % q == target:
                    bad.add(r)
        return bad
    
    # Precompute bad residues for small primes
    # We need q such that the condition is meaningful. 
    # If q > p^2 + max_k, then q cannot divide any p^2 + j, so no constraint.
    # But for pre-sieving over a range of p, we use small q.
    # The key insight: if q is small, and p mod q is in the bad set, then some p^2+j is divisible by q.
    # Since q <= p (for p > q), this doesn't immediately disqualify p (factor q <= p is OK).
    # Wait, the condition is that ALL prime factors must be <= p.
    # So if p^2 + j has a prime factor q > p, it's bad.
    # For small q, q <= p is usually true, so q being a factor is fine.
    # The pre-sieve should target LARGE prime factors.
    # So we should use large q? But large q means few residues.
    
    # Let me reconsider. The condition is: for all j in [1,k], all prime factors of p^2+j are <= p.
    # Equivalently, there is no prime q > p such that q | (p^2 + j) for some j in [1,k].
    # So we want to find p such that no prime q > p divides any of p^2+1, ..., p^2+k.
    
    # For a fixed prime q, p^2 + j ≡ 0 (mod q) means p^2 ≡ -j (mod q).
    # If q > p, then p mod q = p (assuming p < q).
    # So we need p^2 ≡ -j (mod q) for some j.
    # This is a constraint on p mod q.
    
    # For the pre-sieve, we can use primes q in a certain range.
    # If we use small q, the constraint is weak because q <= p is likely.
    # We should use q > p? But p varies.
    
    # Alternative approach: for a given p, check if any p^2+j has a prime factor > p.
    # We can do this by factoring each p^2+j and checking the max factor.
    
    # Let's just use the fast factorization approach with a good search strategy.
    
    # Search strategy:
    # 1. Try large primes first (larger p makes it easier for factors to be <= p)
    # 2. Use the congruence sieve to quickly eliminate candidates
    
    # For the congruence sieve with variable p:
    # For a candidate p, and for each small prime q (say q <= 1000), check if there exists j in [1,MAX_K]
    # such that q > p and q | (p^2 + j). But q <= 1000 < p for large p, so this never happens.
    # So small q are never a problem for large p.
    
    # The real constraint is from large q. But we can't pre-sieve over all large q.
    
    # Let's just do a direct search with fast factorization.
    # Search primes in increasing order, but start from a good range.
    
    best_k = 0
    best_p = 2
    
    # Try a range of primes. Larger primes are better.
    # Let's search in chunks.
    
    search_start = 1000
    search_end = 1000000  # Start with a reasonable range
    
    # Generate primes in range
    primes = primes_in_range(search_start, search_end)
    
    for p in primes:
        if time.time() > deadline - 5:
            break
        # Check run length
        k = 0
        p_sq = p * p
        while k < MAX_K:
            val = p_sq + k + 1
            # Check if all prime factors of val are <= p
            # Quick check: if val is prime and val > p, fail
            if is_prime(val):
                if val > p:
                    break
                else:
                    k += 1
                    continue
            # Factorize and check
            factors = factorize(val)
            if factors and max(factors) > p:
                break
            k += 1
        if k > best_k:
            best_k = k
            best_p = p
            if best_k == MAX_K:
                break
    
    # If we have time, search larger primes
    if time.time() < deadline - 10 and best_k < MAX_K:
        search_start = search_end + 1
        search_end = 10000000
        primes = primes_in_range(search_start, search_end)
        for p in primes:
            if time.time() > deadline - 5:
                break
            k = 0
            p_sq = p * p
            while k < MAX_K:
                val = p_sq + k + 1
                if is_prime(val):
                    if val > p:
                        break
                    else:
                        k += 1
                        continue
                factors = factorize(val)
                if factors and max(factors) > p:
                    break
                k += 1
            if k > best_k:
                best_k = k
                best_p = p
                if best_k == MAX_K:
                    break
    
    return {'p': best_p, 'k': best_k}
