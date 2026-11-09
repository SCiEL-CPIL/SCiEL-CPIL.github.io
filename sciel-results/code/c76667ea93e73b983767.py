"""Rewritten program with faster factorization and broader search."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    deadline = time.time() + 138  # safety margin before 150s kill
    
    # --- Deterministic Miller-Rabin for 64-bit integers ---
    _MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    
    def is_prime_mr(n):
        if n < 2:
            return False
        for p in _MR_BASES:
            if n == p:
                return True
            if n % p == 0:
                return False
        # write n-1 as 2^r * d
        r, d = 0, n - 1
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in _MR_BASES:
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
    def _gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randrange(2, n)
            c = random.randrange(1, n)
            m = 128
            g, q, r = 1, 1, 1
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
                    g = _gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                while True:
                    ys = (ys * ys + c) % n
                    g = _gcd(abs(x - ys), n)
                    if g > 1:
                        break
            if g != n:
                return g
    
    def factorize(n):
        """Return list of prime factors of n (with multiplicity)."""
        factors = []
        # Trial division by small primes first
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        for p in small_primes:
            while n % p == 0:
                factors.append(p)
                n //= p
        if n == 1:
            return factors
        # Recursive factorization
        stack = [n]
        while stack:
            m = stack.pop()
            if m == 1:
                continue
            if is_prime_mr(m):
                factors.append(m)
                continue
            d = pollard_brent(m)
            stack.append(d)
            stack.append(m // d)
        return factors
    
    def largest_prime_factor(n):
        """Return the largest prime factor of n."""
        if n <= 1:
            return 1
        # Quick trial division by small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        m = n
        for p in small_primes:
            while m % p == 0:
                m //= p
        if m == 1:
            return small_primes[-1]
        # Check if m is prime
        if is_prime_mr(m):
            return m
        # Factor m using Pollard-Brent
        factors = factorize(m)
        return max(factors)
    
    # --- Sieve for primes up to a limit ---
    def sieve_primes(limit):
        if limit < 2:
            return []
        bs = bytearray(b'\x01') * (limit + 1)
        bs[0] = bs[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if bs[i]:
                bs[i*i::i] = bytearray(len(bs[i*i::i]))
        return [i for i in range(2, limit + 1) if bs[i]]
    
    # Precompute primes up to 10^6 for trial division pre-filter
    SIEVE_LIMIT = 10**6
    primes_small = sieve_primes(SIEVE_LIMIT)
    
    # For checking if all prime factors of n are <= p, we can first divide out
    # all primes <= p from n; if remainder is 1, all factors are <= p.
    # But we need to be careful: we only need to check if the LARGEST factor <= p.
    # A faster approach: divide n by all primes <= p. If the remaining m > 1,
    # then m has a prime factor > p (since all primes <= p have been removed).
    # But this requires iterating over many primes. Instead, use:
    # If n has a prime factor > p, then n > p^2 (since the smallest such factor
    # combined with any cofactor... actually no, e.g., n = q where q > p).
    # Better: check if n is prime and > p, or if n has a factor in (p, sqrt(n)]...
    
    # Actually the most efficient check: 
    # 1. If n <= p: all factors are <= p trivially (n=1 or n is prime <= p or composite with factors <= p)
    # 2. Divide n by all primes <= min(p, sqrt(n)). If remaining m > 1, m is prime and > p, so fail.
    #    If m == 1, all factors <= p, success.
    # But dividing by all primes up to p when p is large (up to 10^9) is too slow.
    
    # Alternative: use factorization to get the largest prime factor.
    # For numbers near p^2 (which is up to 10^18), factorization with Pollard-Brent
    # should be feasible.
    
    # Let's use a hybrid: first try trial division by primes up to a reasonable limit,
    # then use Pollard-Brent for the remainder.
    
    def all_factors_leq(n, limit):
        """Check if all prime factors of n are <= limit."""
        if n <= 1:
            return True
        if n <= limit:
            return True  # n itself is <= limit, so all its factors are <= limit
        # Try trial division by small primes
        m = n
        for p in primes_small:
            if p > limit:
                break
            if p * p > m:
                break
            while m % p == 0:
                m //= p
            if m == 1:
                return True
        # Now m has no prime factors <= min(limit, sqrt(original n) roughly)
        # If m == 1, done
        if m == 1:
            return True
        # If m is prime:
        if is_prime_mr(m):
            return m <= limit
        # Otherwise, m is composite with all prime factors > primes_small limit or > limit
        # Need to factor m further
        # If m <= limit * limit, then it could have factors up to m
        # Let's just factor m and check
        factors = factorize(m)
        return all(f <= limit for f in factors)
    
    # --- Search ---
    best_k = 0
    best_p = 2
    
    # First, search primes up to SIEVE_LIMIT
    for p in primes_small:
        if time.time() > deadline:
            break
        p2 = p * p
        k = 0
        while k < 200:
            n = p2 + k + 1
            if not all_factors_leq(n, p):
                break
            k += 1
            if time.time() > deadline:
                break
        if k > best_k:
            best_k = k
            best_p = p
    
    # Now search larger primes up to 10^9
    # Generate primes in ranges using a segmented sieve or just test candidates
    # For efficiency, we'll test odd numbers and use Miller-Rabin
    
    # Search strategy: test primes in increasing order, but skip ahead if time is running low
    # We can also try random primes in the larger range
    
    # First, let's search primes from SIEVE_LIMIT+1 up to some upper bound
    # We'll use a simple approach: test odd numbers, use MR primality test
    
    search_start = SIEVE_LIMIT + 1
    if search_start % 2 == 0:
        search_start += 1
    
    # We'll search up to 10^9 but with a time budget
    # To make this efficient, we can batch-test ranges
    
    # Let's search in chunks
    chunk_size = 100000
    current = search_start
    max_search = 10**9
    
    while current < max_search and time.time() < deadline:
        # Process a chunk of candidates
        chunk_end = min(current + chunk_size, max_search)
        for cand in range(current, chunk_end, 2):  # odd numbers only
            if time.time() > deadline:
                break
            if not is_prime_mr(cand):
                continue
            p = cand
            p2 = p * p
            k = 0
            while k < 200:
                n = p2 + k + 1
                if not all_factors_leq(n, p):
                    break
                k += 1
                if time.time() > deadline:
                    break
            if k > best_k:
                best_k = k
                best_p = p
        current = chunk_end
    
    # If we still have time, try some random large primes
    if time.time() < deadline - 10:
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            cand = random.randrange(10**8, 10**9)
            if cand % 2 == 0:
                cand += 1
            # Find next prime
            for _ in range(1000):
                if is_prime_mr(cand):
                    p = cand
                    p2 = p * p
                    k = 0
                    while k < 200:
                        n = p2 + k + 1
                        if not all_factors_leq(n, p):
                            break
                        k += 1
                        if time.time() > deadline - 5:
                            break
                    if k > best_k:
                        best_k = k
                        best_p = p
                    break
                cand += 2
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
