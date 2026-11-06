import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Precompute primes up to 10^6 for trial division
    limit = 1000000
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    small_primes = [i for i in range(2, limit + 1) if sieve[i]]
    
    # Deterministic Miller-Rabin for n < 2^64
    _MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    _SMALL_PRIMES_MR = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
                        41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    def is_prime(n):
        if n < 2:
            return False
        for p in _SMALL_PRIMES_MR:
            if n == p:
                return True
            if n % p == 0:
                return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in _MR_BASES:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True
    
    # Pollard-Brent
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randrange(1, n)
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
    
    def is_p_smooth(n, p):
        """Check if all prime factors of n are <= p."""
        if n <= 1:
            return True
        # Trial divide by small primes up to min(p, limit)
        temp = n
        for q in small_primes:
            if q > p or q * q > temp:
                break
            if temp % q == 0:
                while temp % q == 0:
                    temp //= q
                if temp == 1:
                    return True
        if temp == 1:
            return True
        # temp has no prime factors <= min(p, limit)
        if temp <= p:
            # All prime factors of temp are <= temp <= p
            return True
        # temp > p and has no small factors
        if is_prime(temp):
            return False  # temp is prime > p
        # temp is composite > p, factor it
        d = pollard_brent(temp)
        # Check d and temp//d
        if d > p or temp // d > p:
            # Need to check if these have factors > p
            # Recursively check
            return is_p_smooth(d, p) and is_p_smooth(temp // d, p)
        return True
    
    best_k = 0
    best_p = 2
    
    # Search large primes: try random candidates in [10^8, 10^9]
    # Quick reject: check i=1..5 first
    num_candidates = 5000
    tried = set()
    
    for _ in range(num_candidates):
        if time.time() > deadline:
            break
        # Random odd number in [10^8, 10^9]
        while True:
            cand = random.randrange(100_000_000, 1_000_000_000, 2)
            if cand not in tried:
                tried.add(cand)
                break
        if not is_prime(cand):
            continue
        p = cand
        base = p * p
        # Quick reject: check i=1..5
        quick_pass = True
        for i in range(1, 6):
            if time.time() > deadline:
                quick_pass = False
                break
            if not is_p_smooth(base + i, p):
                quick_pass = False
                break
        if not quick_pass:
            continue
        # Full check
        k = 0
        for i in range(1, 201):
            if time.time() > deadline:
                break
            if not is_p_smooth(base + i, p):
                break
            k += 1
        if k > best_k:
            best_k = k
            best_p = p
            if best_k == 200:
                return {'p': p, 'k': 200}
    
    return {'p': best_p, 'k': best_k}
