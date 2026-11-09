"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math

def _is_prime_mr(n):
    """Deterministic Miller-Rabin for n < 3.3e24."""
    if n < 2:
        return False
    if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        return True
    if n % 2 == 0:
        return False
    # write n-1 = d*2^r
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if a >= n:
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

def _pollard_brent(n):
    """Pollard's Brent cycle detection for finding a factor of n."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randint(2, n - 1)
        c = random.randint(1, n - 1)
        m = random.randint(1, n - 1)
        g = r = q = 1
        x = ys = y
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

def _factor(n, primes=None):
    """Full factorization of n. Returns dict {prime: exponent}."""
    if n <= 1:
        return {}
    result = {}
    # trial division by small primes
    if primes:
        for p in primes:
            if p * p > n:
                break
            if n % p == 0:
                cnt = 0
                while n % p == 0:
                    n //= p
                    cnt += 1
                result[p] = cnt
    if n > 1:
        if _is_prime_mr(n):
            result[n] = result.get(n, 0) + 1
        else:
            # use Pollard-Brent
            stack = [n]
            while stack:
                m = stack.pop()
                if m == 1:
                    continue
                if _is_prime_mr(m):
                    result[m] = result.get(m, 0) + 1
                else:
                    d = _pollard_brent(m)
                    stack.append(d)
                    stack.append(m // d)
    return result

def _sieve(limit):
    """Sieve of Eratosthenes, returns list of primes up to limit."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    # Precompute small primes for trial division
    SIEVE_LIMIT = 100000
    small_primes = _sieve(SIEVE_LIMIT)

    best_k = 0
    best_p = 2

    # Phase 1: Search small primes (up to ~50000) with fast checking
    # For small p, we can check p^2+1 ... p^2+k quickly
    for p in small_primes:
        if p > 50000:
            break
        if time.time() - start_time > DEADLINE:
            break
        p2 = p * p
        k = 0
        for i in range(1, 201):
            if time.time() - start_time > DEADLINE:
                break
            n = p2 + i
            # Check if all prime factors of n are <= p
            # Quick check: divide out small primes
            tmp = n
            valid = True
            for sp in small_primes:
                if sp > p:
                    break
                if sp * sp > tmp:
                    break
                while tmp % sp == 0:
                    tmp //= sp
            # Now tmp should be 1 or a prime > p (or composite with factors > p)
            if tmp > 1:
                # tmp has a factor > p (itself or a factor of it)
                # Check if tmp is prime and > p, or has a factor > p
                if tmp <= p:
                    # all factors were <= p
                    valid = True
                else:
                    valid = False
            if not valid:
                break
            k += 1
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Search larger primes with smarter strategy
    # Use random sampling of larger primes and structured search
    # Key insight: for large p, p^2+i is ~p^2, so factors are ~p
    # We need all factors <= p, meaning p^2+i must be p-smooth
    
    # Try primes in ranges, using a faster "is p-smooth" check
    # A number n is p-smooth if after removing all prime factors <= p, remainder is 1
    
    # For larger p, use random sampling in strategic ranges
    random.seed(42)
    
    # Try specific ranges where long runs might occur
    # Based on number theory, larger p tends to have longer runs
    search_ranges = [
        (50000, 200000),
        (200000, 1000000),
        (1000000, 10000000),
        (10000000, 100000000),
        (100000000, 1000000000),
    ]
    
    for lo, hi in search_ranges:
        if time.time() - start_time > DEADLINE:
            break
        # Sample primes in this range
        # Use a simple approach: generate candidates and test primality
        num_tried = 0
        max_tries = 500  # limit per range
        while num_tried < max_tries and time.time() - start_time < DEADLINE:
            # Generate a random odd number in range
            if hi - lo < 1000:
                p = random.randint(lo, hi)
            else:
                p = random.randint(lo, hi)
            if p % 2 == 0:
                p += 1
            if p < 2:
                continue
            num_tried += 1
            if not _is_prime_mr(p):
                continue
            if p > 10**9:
                continue
            # Check p-smoothness of p^2+1 ... p^2+k
            p2 = p * p
            k = 0
            for i in range(1, 201):
                if time.time() - start_time > DEADLINE:
                    break
                n = p2 + i
                # Fast p-smooth check using trial division by small primes
                # then Pollard-Brent for remaining
                tmp = n
                # Remove small prime factors
                for sp in small_primes:
                    if sp > p:
                        break
                    if sp * sp > tmp:
                        break
                    while tmp % sp == 0:
                        tmp //= sp
                if tmp > 1:
                    # Need to check if tmp has any factor > p
                    # If tmp is prime: it's a factor, must be <= p
                    # If tmp is composite: all its prime factors must be <= p
                    if _is_prime_mr(tmp):
                        if tmp > p:
                            break
                    else:
                        # Factor tmp partially to check max factor
                        # Use Pollard-Brent
                        factors = _factor(tmp, small_primes)
                        if factors:
                            max_f = max(factors.keys())
                            if max_f > p:
                                break
                        else:
                            # tmp was fully factored by small primes
                            pass
                k += 1
            if k > best_k:
                best_k = k
                best_p = p

    # Phase 3: If we have time, do targeted search around promising p values
    # Also try sequential search near best_p
    if time.time() - start_time < DEADLINE - 10:
        # Search near best_p
        for offset in range(-1000, 1001):
            if time.time() - start_time > DEADLINE:
                break
            p = best_p + offset
            if p < 2 or p > 10**9:
                continue
            if not _is_prime_mr(p):
                continue
            p2 = p * p
            k = 0
            for i in range(1, 201):
                if time.time() - start_time > DEADLINE:
                    break
                n = p2 + i
                tmp = n
                for sp in small_primes:
                    if sp > p:
                        break
                    if sp * sp > tmp:
                        break
                    while tmp % sp == 0:
                        tmp //= sp
                if tmp > 1:
                    if _is_prime_mr(tmp):
                        if tmp > p:
                            break
                    else:
                        factors = _factor(tmp, small_primes)
                        if factors:
                            max_f = max(factors.keys())
                            if max_f > p:
                                break
                k += 1
            if k > best_k:
                best_k = k
                best_p = p

    if best_k < 1:
        best_k = 1
        best_p = 2

    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
