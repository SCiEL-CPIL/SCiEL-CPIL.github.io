import time
import random
import math
import sys

# Deterministic Miller-Rabin bases for n < 3,317,044,064,679,887,385,961,981
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_prime(n):
    if n < 2:
        return False
    # Small primes check
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    # Miller-Rabin
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_BASES:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randint(2, n - 1)
        c = random.randint(1, n - 1)
        m = random.randint(1, 100)
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
                    q = (q * abs(x - y)) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g != 1:
                    break
        if g != n:
            return g

def factor(n, factors=None):
    if factors is None:
        factors = []
    if n <= 1:
        return factors
    if is_prime(n):
        factors.append(n)
        return factors
    # Try small primes first
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if n % p == 0:
            factors.append(p)
            return factor(n // p, factors)
    d = pollard_brent(n)
    factors.append(d)
    return factor(n // d, factors)

def largest_prime_factor(n):
    """Return the largest prime factor of n."""
    if n <= 1:
        return 1
    # Trial divide by small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        while n % p == 0:
            n //= p
    if n == 1:
        return small_primes[-1]  # largest small prime that divided it, but actually we need to track
    # Let's redo: we want the largest prime factor
    # Better approach: factor fully and take max
    return _largest_pf(n)

def _largest_pf(n):
    """Compute largest prime factor of n using trial division + Pollard-Brent."""
    if n <= 1:
        return 1
    max_pf = 1
    # Trial division by small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        while n % p == 0:
            max_pf = p
            n //= p
    if n == 1:
        return max_pf
    # Now n has no small prime factors. Use Pollard-Brent to factor it.
    # We need to find all prime factors and take the max.
    def _factor_into_primes(m):
        if m <= 1:
            return []
        if is_prime(m):
            return [m]
        d = pollard_brent(m)
        return _factor_into_primes(d) + _factor_into_primes(m // d)
    
    primes = _factor_into_primes(n)
    for p in primes:
        if p > max_pf:
            max_pf = p
    return max_pf

def has_all_factors_at_most_p(n, p):
    """Check if all prime factors of n are <= p."""
    if n <= 1:
        return True
    # Trial divide by small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for sp in small_primes:
        if sp > p:
            # If sp > p and n is divisible by sp, then n has a factor > p
            # But sp might not divide n. We just need to check if n has any factor > p.
            # Actually if sp > p, we can't use it for trial division in this context.
            # But we've already divided by all small primes <= p (or all small primes if p >= 37).
            pass
        while n % sp == 0:
            n //= sp
    if n == 1:
        return True
    # Now n has no small prime factors. If n > 1, we need to check its prime factors.
    # If n itself is prime and n > p, return False.
    # If n is composite, we need to factor it.
    if is_prime(n):
        return n <= p
    # n is composite with no small prime factors
    # Factor it
    def _factor_into_primes(m):
        if m <= 1:
            return []
        if is_prime(m):
            return [m]
        d = pollard_brent(m)
        return _factor_into_primes(d) + _factor_into_primes(m // d)
    
    for pf in _factor_into_primes(n):
        if pf > p:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 100  # safety margin: evaluator kills at 150s
    
    best_k = 0
    best_p = 3
    
    # Precompute small primes up to 10000 for trial division
    limit = 10000
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit+1, i):
                sieve[j] = False
    small_primes_list = [i for i in range(2, limit+1) if sieve[i]]
    
    def is_prime_fast(n):
        if n < 2:
            return False
        if n < limit:
            return sieve[n]
        # Check against small primes
        for sp in small_primes_list:
            if sp * sp > n:
                return True
            if n % sp == 0:
                return False
        # Miller-Rabin
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in _MR_BASES:
            if a % n == 0:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True
    
    def all_factors_leq(n, p):
        """Check all prime factors of n are <= p. Returns True if all factors <= p."""
        if n <= 1:
            return True
        # Trial divide by small primes up to min(p, limit)
        for sp in small_primes_list:
            if sp > p:
                break
            if sp * sp > n:
                break
            while n % sp == 0:
                n //= sp
        if n == 1:
            return True
        # If n is prime, check if it's <= p
        if is_prime_fast(n):
            return n <= p
        # n is composite, factor it and check all factors
        def _fp(m):
            if m <= 1:
                return []
            if is_prime_fast(m):
                return [m]
            d = pollard_brent(m)
            return _fp(d) + _fp(m // d)
        for pf in _fp(n):
            if pf > p:
                return False
        return True
    
    # Quick scan of small primes to get a baseline
    small_primes_to_check = [3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499]
    
    for p in small_primes_to_check:
        if time.time() > deadline:
            break
        base = p * p
        k = 0
        for i in range(1, 201):
            if time.time() > deadline:
                break
            n = base + i
            if all_factors_leq(n, p):
                k = i
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
    
    # Sample medium and large primes
    random.seed(42)
    attempts = 0
    while time.time() < deadline - 10:
        attempts += 1
        # Generate candidate prime
        if attempts % 5 == 0:
            p = random.randint(10**5, 10**6) | 1
        elif attempts % 5 == 1:
            p = random.randint(10**6, 10**7) | 1
        elif attempts % 5 == 2:
            p = random.randint(10**7, 10**8) | 1
        else:
            p = random.randint(10**8, 10**9) | 1
        
        if not is_prime_fast(p):
            continue
        
        # Check the run for this p
        base = p * p
        k = 0
        for i in range(1, 201):
            if time.time() > deadline - 5:
                break
            n = base + i
            if all_factors_leq(n, p):
                k = i
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
            # If we found a good k, try nearby primes
            if k >= 10 and time.time() < deadline - 20:
                for dp in range(2, 20, 2):
                    if time.time() > deadline - 5:
                        break
                    p2 = p + dp
                    if is_prime_fast(p2):
                        base2 = p2 * p2
                        k2 = 0
                        for i in range(1, 201):
                            if time.time() > deadline - 5:
                                break
                            n2 = base2 + i
                            if all_factors_leq(n2, p2):
                                k2 = i
                            else:
                                break
                        if k2 > best_k:
                            best_k = k2
                            best_p = p2
    
    return {'p': best_p, 'k': best_k}
