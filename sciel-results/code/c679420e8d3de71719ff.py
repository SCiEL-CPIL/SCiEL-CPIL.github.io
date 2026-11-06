import math
import time
import random

# Deterministic Miller-Rabin for n < 2^64
def is_prime(n):
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
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
    
    # Deterministic witnesses for n < 2^64
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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

def pollard_rho(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    
    # Use a deterministic seed for reproducibility if needed, but random is fine
    x = random.randrange(2, n)
    y = x
    c = random.randrange(1, n)
    d = 1
    
    while d == 1:
        x = (x * x + c) % n
        y = (y * y + c) % n
        y = (y * y + c) % n
        d = math.gcd(abs(x - y), n)
        
    if d != n:
        return d
    else:
        return pollard_rho(n)

def _gen_primes(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

# Generate primes up to 10^6 for trial division
MAX_TRIAL_PRIME = 100_000
PRIMES = _gen_primes(MAX_TRIAL_PRIME)

def is_p_smooth(n, p):
    if n <= p:
        return True
    
    m = n
    # Trial division
    for q in PRIMES:
        if q > p:
            # If we've tried all primes up to p, and m > 1, then m has a factor > p
            # UNLESS m is 1. But if m > 1 and no factor <= p, then m is not p-smooth.
            # However, we might break earlier if q*q > m.
            # If q > p, we stop. If m > 1, return False.
            return m == 1
        
        if q * q > m:
            # m is prime
            return m <= p
        
        while m % q == 0:
            m //= q
        
        if m == 1:
            return True
        if m <= p:
            # m is prime (since we divided out all factors < sqrt(m) or up to p)
            # Actually if m <= p and we haven't found factors, m is prime.
            return True

    # If we exhausted PRIMES (up to 10^5)
    if m == 1:
        return True
    
    # m is > 1 and has no prime factors <= 10^5.
    # Check if m is prime
    if is_prime(m):
        return m <= p
    
    # m is composite. Factorize it.
    # Since m has no small factors, Pollard Rho should be fast.
    # We need to check if all prime factors of m are <= p.
    # Note: m < n = p^2 + i.
    # If m has a factor > p, it's not smooth.
    
    # Recursive factorization
    def check(m):
        if m == 1:
            return True
        if m <= p:
            return True
        if is_prime(m):
            return m <= p
        
        d = pollard_rho(m)
        return check(d) and check(m // d)
        
    return check(m)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_k = 0
    best_p = 2
    
    # Generate primes for search up to a reasonable limit
    # We can't go up to 10^9.
    # Let's try primes up to 10^7 first.
    SEARCH_PRIMES = _gen_primes(10_000_000)
    
    for p in SEARCH_PRIMES:
        if time.time() > deadline:
            break
            
        p_sq = p * p
        k = 0
        
        for i in range(1, 201):
            if time.time() > deadline:
                break
                
            n_val = p_sq + i
            if is_p_smooth(n_val, p):
                k = i
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p
            
        if best_k == 200:
            break

    return {'p': best_p, 'k': best_k}
