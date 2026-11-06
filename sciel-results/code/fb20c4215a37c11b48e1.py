import math
import random
import time

# Deterministic Miller-Rabin for n < 3.3e24
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_prime_mr(n):
    if n < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
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

# Pollard-Brent
def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randrange(2, n - 1)
        c = random.randrange(1, n - 1)
        m = 128
        g = r = q = 1
        x = ys = 0
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

# Factor a number, returning a list of prime factors
def factorize(n):
    factors = []
    if n <= 1:
        return factors
    if is_prime_mr(n):
        factors.append(n)
        return factors
    
    # Trial division by small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        while n % p == 0:
            factors.append(p)
            n //= p
    if n == 1:
        return factors
    if is_prime_mr(n):
        factors.append(n)
        return factors
        
    d = pollard_brent(n)
    factors.extend(factorize(d))
    factors.extend(factorize(n // d))
    return factors

def is_p_smooth(n, p, small_primes):
    """
    Check if n is p-smooth.
    Returns True if all prime factors of n are <= p.
    """
    if n <= 1:
        return True
    
    # Quick check: if n is prime, it's only p-smooth if n <= p
    if is_prime_mr(n):
        return n <= p
    
    # Trial division by small primes
    temp_n = n
    for q in small_primes:
        if q > p:
            break
        if temp_n % q == 0:
            while temp_n % q == 0:
                temp_n //= q
        if temp_n == 1:
            return True
        if temp_n <= p:
            # If remaining is <= p, it must be p-smooth? 
            # Not necessarily, it could be a prime > p? No, if temp_n <= p, 
            # and we've divided out all small primes, temp_n is either 1 or a prime.
            # If it's a prime, it's <= p, so it's fine.
            # If it's composite, its factors are < sqrt(temp_n) <= sqrt(p) < p.
            # So yes, if temp_n <= p, it is p-smooth.
            return True
            
    if temp_n == 1:
        return True
        
    # If temp_n > p, we need to factor it
    # But first, check if temp_n is prime
    if is_prime_mr(temp_n):
        return temp_n <= p
        
    # Factor the remainder
    factors = factorize(temp_n)
    for f in factors:
        if f > p:
            return False
    return True

def sieve(limit):
    if limit < 2:
        return []
    bs = bytearray([1]) * (limit + 1)
    bs[0] = bs[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if bs[i]:
            step = i
            start = i * i
            bs[start::step] = bytearray(len(bs[start::step]))
    return [i for i in range(2, limit + 1) if bs[i]]

def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # Safety margin
    random.seed(42)
    
    # Precompute small primes for trial division
    small_primes = sieve(1000)
    
    best_k = 0
    best_p = 2
    
    # Phase 1: Scan primes up to 100,000
    # This is fast and covers a good range.
    primes_to_scan = sieve(100000)
    
    for p in primes_to_scan:
        if time.time() >= deadline - 2.0:
            break
        base = p * p
        k = 0
        for j in range(1, 201):
            if time.time() >= deadline - 2.0:
                break
            n = base + j
            if not is_p_smooth(n, p, small_primes):
                break
            k += 1
        if k > best_k:
            best_k = k
            best_p = p
            if best_k == 200:
                return {'p': best_p, 'k': best_k}

    # Phase 2: Random search for larger primes
    # Larger p makes it easier to be p-smooth.
    # We will sample random numbers and check if they are prime.
    
    # Ranges to sample
    ranges = [
        (100000, 1000000),
        (1000000, 10000000),
        (10000000, 100000000),
        (100000000, 1000000000)
    ]
    
    for low, high in ranges:
        if time.time() >= deadline - 5.0:
            break
        
        # Estimate number of attempts
        # We want to spend roughly (deadline - current_time) / time_per_attempt
        # Let's just loop until deadline
        attempts = 0
        while time.time() < deadline - 3.0:
            attempts += 1
            # Pick a random odd number
            n = random.randrange(low, high)
            if n % 2 == 0:
                n += 1
            if n > 10**9:
                n = 10**9 - 1 - (n % 10) # Just clamp, though range handles it
            
            # Check if prime
            if not is_prime_mr(n):
                continue
            
            p = n
            base = p * p
            k = 0
            for j in range(1, 201):
                if time.time() >= deadline - 3.0:
                    break
                num = base + j
                if not is_p_smooth(num, p, small_primes):
                    break
                k += 1
            if k > best_k:
                best_k = k
                best_p = p
                if best_k == 200:
                    return {'p': best_p, 'k': best_k}
                
            # If k is 0, it was very cheap. We can do more attempts.
            # If k is high, it was expensive.
            # No dynamic adjustment needed, just run until deadline.

    return {'p': best_p, 'k': best_k}
