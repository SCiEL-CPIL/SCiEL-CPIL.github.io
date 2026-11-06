import time
import math
import random
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

def sieve(n):
    """Return a list of primes up to n using bytearray."""
    if n < 2:
        return []
    is_prime_arr = bytearray(b'\x01') * (n + 1)
    is_prime_arr[0] = is_prime_arr[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if is_prime_arr[i]:
            is_prime_arr[i*i::i] = bytearray(len(is_prime_arr[i*i::i]))
    return [i for i in range(n+1) if is_prime_arr[i]]

def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    
    y, c, m = random.randint(1, n-1), random.randint(1, n-1), 128
    g, r, q = 1, 1, 1
    while g == 1:
        x = y
        for i in range(r):
            y = (y*y + c) % n
        k = 0
        while k < r and g == 1:
            ys = y
            for i in range(min(m, r-k)):
                y = (y*y + c) % n
                q = (q * abs(x-y)) % n
            g = math.gcd(q, n)
            k += m
        r *= 2
    if g == n:
        while True:
            ys = (ys*ys + c) % n
            g = math.gcd(abs(x-ys), n)
            if g != 1:
                break
    return g

_SMALL_PRIMES_LIMIT = 100000
_small_primes = None

def _is_p_smooth(n, p, small_primes):
    """
    Check if all prime factors of n are <= p.
    Optimized for n ~ p^2.
    """
    if n <= 1:
        return True
    if n <= p:
        return True
    
    # Trial divide by small primes
    rem = n
    for q in small_primes:
        if q > p:
            break
        if q * q > rem:
            break
        while rem % q == 0:
            rem //= q
            
    if rem == 1:
        return True
        
    if rem <= p:
        return True
        
    # rem > p
    if is_prime(rem):
        return False
        
    # rem is composite and > p
    # We need to check if rem has any factor > p
    # Use Pollard-Brent to find a factor
    d = pollard_brent(rem)
    # Check d and rem//d
    return _is_p_smooth(d, p, small_primes) and _is_p_smooth(rem // d, p, small_primes)

def construct():
    global _small_primes
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    if _small_primes is None:
        _small_primes = sieve(_SMALL_PRIMES_LIMIT)
        
    best_k = 0
    best_p = 2
    
    # 1. Check small primes first (up to 100,000)
    # This is fast and might find a good candidate
    small_check_primes = sieve(100000)
    for p in small_check_primes:
        if time.time() > deadline - 10:
            break
        base = p * p
        k = 0
        for i in range(1, 201):
            if time.time() > deadline - 10:
                break
            n = base + i
            if _is_p_smooth(n, p, _small_primes):
                k = i
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p

    # 2. Random sampling of larger primes
    # We want to sample primes in [10^6, 10^9]
    # To generate a prime, we pick a random odd number and test.
    # We can bias towards numbers that are not divisible by small primes.
    
    # Wheel for 2, 3, 5: offsets 1, 7, 11, 13, 17, 19, 23, 29
    wheel = [1, 7, 11, 13, 17, 19, 23, 29]
    
    while time.time() < deadline - 5:
        # Pick a random base and offset
        base_num = random.randint(10**6, 10**9)
        offset = random.choice(wheel)
        p_candidate = base_num + offset
        
        # Ensure range
        if p_candidate < 10**6:
            p_candidate += 30
        if p_candidate > 10**9:
            p_candidate -= 30
            
        # Quick primality test
        if not is_prime(p_candidate):
            continue
            
        p = p_candidate
        base = p * p
        k = 0
        for i in range(1, 201):
            if time.time() > deadline - 5:
                break
            n = base + i
            if _is_p_smooth(n, p, _small_primes):
                k = i
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p
            
    return {'p': best_p, 'k': best_k}
