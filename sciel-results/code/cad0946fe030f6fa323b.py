import time
import math
import random
from functools import lru_cache

def _is_prime_mr(n):
    """Deterministic Miller-Rabin for n < 3.3e24."""
    if n < 2:
        return False
    if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        return True
    if n % 2 == 0:
        return False
    
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
    if n % 5 == 0:
        return 5
    
    # Use deterministic sequence for reproducibility and speed
    y = random.randint(1, n - 1)
    c = random.randint(1, n - 1)
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
                q = (q * abs(x - y)) % n
            g = math.gcd(q, n)
            k += m
        r *= 2
        
    if g == n:
        while True:
            ys = (ys * ys + c) % n
            g = math.gcd(abs(x - ys), n)
            if g > 1:
                break
                
    return g

def _sieve(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, limit + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    # Precompute small primes for trial division - keep limit modest for speed
    small_primes = _sieve(10000)  # 1229 primes up to 10^4
    
    def is_p_smooth(n, p):
        """Check if n is p-smooth using small_primes and Pollard-Brent."""
        # Trial division by small primes
        for q in small_primes:
            if q > p:
                break
            if q * q > n:
                break
            while n % q == 0:
                n //= q
        
        if n == 1:
            return True
            
        if _is_prime_mr(n):
            return n <= p
            
        # n is composite and > 1
        # Find a factor
        try:
            f = _pollard_brent(n)
        except Exception:
            return False
        
        return is_p_smooth(f, p) and is_p_smooth(n // f, p)

    best_k = 0
    best_p = 2
    
    # Generate primes for search
    # Search in a more targeted way: focus on primes where we expect longer runs
    search_primes = _sieve(2000000)  # Primes up to 2*10^6
    
    # Phase 1: Scan primes, but skip some for speed
    for idx, p in enumerate(search_primes):
        if time.time() > deadline:
            break
        
        # Skip some primes to allow more per prime check
        if idx % 3 != 0 and p > 1000:
            continue
            
        k = 0
        base = p * p
        while k < 200:
            if time.time() > deadline:
                break
            n = base + k + 1
            if is_p_smooth(n, p):
                k += 1
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Try some larger primes with random sampling
    if time.time() < deadline - 15:
        attempts = 0
        while time.time() < deadline - 5 and attempts < 2000:
            attempts += 1
            # Generate random odd number in [2*10^6, 10^9]
            p_candidate = random.randint(2 * 10**6, 10**9)
            if p_candidate % 2 == 0:
                p_candidate += 1
                
            if _is_prime_mr(p_candidate):
                k = 0
                base = p_candidate * p_candidate
                while k < 200:
                    if time.time() > deadline:
                        break
                    n = base + k + 1
                    if is_p_smooth(n, p_candidate):
                        k += 1
                    else:
                        break
                        
                if k > best_k:
                    best_k = k
                    best_p = p_candidate

    # Ensure we have a valid result
    if best_p == 0:
        best_p = 2
        best_k = 1
    
    return {'p': best_p, 'k': best_k}
