"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
import time

def is_prime_miller_rabin(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
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
        y = random.randrange(1, n)
        c = random.randrange(1, n)
        m = 128
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
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g > 1:
                    break
        if g != n:
            return g

def is_p_smooth(n, p, small_primes):
    """Check if n is p-smooth (all prime factors <= p)."""
    if n <= 1:
        return True
    n_rem = n
    # Trial division by small primes
    for q in small_primes:
        if q > p or q * q > n_rem:
            break
        while n_rem % q == 0:
            n_rem //= q
    if n_rem == 1:
        return True
    # Check if remainder is prime
    if is_prime_miller_rabin(n_rem):
        return n_rem <= p
    # If composite, find a factor
    # Since n < p^2 + 200, and we removed small factors,
    # any factor of n_rem is > small_primes[-1] (approx 10^6)
    # If n_rem has a factor > p, it's not p-smooth.
    # We can try to find a factor.
    # If we find a factor f:
    # If f > p, return False.
    # Else, check if n_rem/f is p-smooth.
    f = pollard_brent(n_rem)
    if f > p:
        return False
    return is_p_smooth(n_rem // f, p, small_primes)

def construct():
    # Generate primes up to 10^6 for trial division
    limit = 1000000
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    small_primes = [i for i in range(2, limit + 1) if sieve[i]]
    
    best_k = 0
    best_p = 2
    
    # Generate candidate primes p
    # We search up to 10^7 to balance time and coverage
    p_limit = 10000000
    # Sieve for primes up to p_limit
    sieve_p = [True] * (p_limit + 1)
    sieve_p[0] = sieve_p[1] = False
    for i in range(2, int(p_limit**0.5) + 1):
        if sieve_p[i]:
            for j in range(i*i, p_limit + 1, i):
                sieve_p[j] = False
    candidate_primes = [i for i in range(2, p_limit + 1) if sieve_p[i]]
    
    # To speed up, we can skip some primes if we are running out of time
    # But let's try to check all primes up to 10^7 first.
    # If that's too slow, we'll optimize.
    
    start_time = time.time()
    time_limit = 110  # Leave 10s margin
    
    for p in candidate_primes:
        if time.time() - start_time > time_limit:
            break
            
        k = 0
        for j in range(1, 201):
            n = p * p + j
            if is_p_smooth(n, p, small_primes):
                k = j
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
            if best_k >= 200:
                break
        
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
