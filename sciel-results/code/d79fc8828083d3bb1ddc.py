import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    # Deterministic Miller-Rabin for 64-bit integers
    def is_prime(n):
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        # Write n-1 as d*2^r
        d, r = n - 1, 0
        while d % 2 == 0:
            d //= 2
            r += 1
        # Bases sufficient for n < 2^64
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
    
    # Pollard's Brent cycle detection
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(2, n - 1)
            c = random.randint(1, n - 1)
            m = random.randint(1, 100)
            g = r = q = 1
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
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
            if g != n:
                return g
    
    def factorize(n):
        """Return list of prime factors (with multiplicity) of n."""
        if n <= 1:
            return []
        factors = []
        # Trial division for small factors
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            while n % p == 0:
                factors.append(p)
                n //= p
        if n == 1:
            return factors
        stack = [n]
        while stack:
            m = stack.pop()
            if m == 1:
                continue
            if is_prime(m):
                factors.append(m)
            else:
                d = pollard_brent(m)
                stack.append(d)
                stack.append(m // d)
        return factors
    
    def largest_prime_factor(n):
        if n <= 1:
            return 1
        # Quick check for small factors
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                n //= p
                while n % p == 0:
                    n //= p
        if n == 1:
            return 1
        factors = factorize(n)
        return max(factors) if factors else 1
    
    def check_k(p, max_k=200):
        """Find the maximum k such that p^2+1..p^2+k all have all prime factors <= p."""
        k = 0
        for i in range(1, max_k + 1):
            n = p * p + i
            # Check if all prime factors of n are <= p
            # Trial divide by primes <= p, but that's expensive for large p
            # Instead, factorize and check max factor
            lf = largest_prime_factor(n)
            if lf > p:
                break
            k += 1
        return k
    
    best_k = 0
    best_p = 2
    
    # Generate primes up to some limit for trial division in is_prime
    # We'll search over primes p in ranges
    
    # First, try small primes quickly
    def primes_upto(n):
        if n < 2:
            return []
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j] = False
        return [i for i in range(2, n+1) if sieve[i]]
    
    small_primes = primes_upto(10000)
    
    # Check small primes
    for p in small_primes:
        if time.time() > deadline:
            break
        k = check_k(p)
        if k > best_k:
            best_k = k
            best_p = p
    
    # For larger primes, use a smarter search
    # Focus on ranges where p is larger, as p-smooth numbers might cluster
    # Try primes in geometrically spaced ranges
    
    search_ranges = [
        (10000, 50000),
        (50000, 200000),
        (200000, 1000000),
        (1000000, 5000000),
        (5000000, 50000000),
        (50000000, 500000000),
        (500000000, 1000000000),
    ]
    
    for lo, hi in search_ranges:
        if time.time() > deadline:
            break
        # Sample primes in this range
        # Use a step to limit the number of candidates
        step = max(1, (hi - lo) // 10000)  # at most ~10000 candidates per range
        p = lo
        if p % 2 == 0:
            p += 1
        while p <= hi and time.time() < deadline:
            if is_prime(p):
                k = check_k(p)
                if k > best_k:
                    best_k = k
                    best_p = p
            p += 2 * step
            if p % 2 == 0:
                p += 1
    
    return {'p': best_p, 'k': best_k}
