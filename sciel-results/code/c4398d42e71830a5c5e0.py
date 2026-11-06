import time
import math
import random

def sieve(n):
    is_prime = bytearray(b'\x01') * (n + 1)
    is_prime[0:2] = b'\x00\x00'
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i:n+1:i] = bytearray(len(is_prime[i*i:n+1:i]))
    return [i for i, prime in enumerate(is_prime) if prime]

# Sieve up to 10^7
_SIEVE_LIMIT = 10000000
_PRIMES = sieve(_SIEVE_LIMIT)
_PRIME_SET = set(_PRIMES)

# Miller-Rabin for 64-bit
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_prime(n):
    if n < 2:
        return False
    for p in _MR_BASES:
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

def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randint(2, n - 1)
        c = random.randint(1, n - 1)
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

def is_p_smooth_individual(n, p):
    """Check if n is p-smooth using trial division and Pollard-Brent."""
    if n <= 1:
        return True
    # Trial division by small primes
    for q in _PRIMES:
        if q > p:
            break
        if n % q == 0:
            while n % q == 0:
                n //= q
    if n == 1:
        return True
    if is_prime(n):
        return n <= p
    # Composite, use Pollard-Brent
    stack = [n]
    while stack:
        x = stack.pop()
        if x == 1:
            continue
        if is_prime(x):
            if x > p:
                return False
            continue
        r = int(math.isqrt(x))
        if r * r == x:
            stack.append(r)
            continue
        d = pollard_brent(x)
        stack.append(d)
        stack.append(x // d)
    return True

def check_window_smooth_sieve(p, k_max=200):
    """
    Check if p^2+1 .. p^2+k_max are all p-smooth.
    Uses a sieve approach for efficiency when p is not too large.
    Returns the max k found.
    """
    # Only use sieve method if p is within our precomputed primes range 
    # or if p is small enough that the window sieve is fast.
    # If p > 10^6, the number of primes to iterate is large.
    # Let's use this for p <= 10^6.
    # For p > 10^6, we fall back to individual checks in the main loop.
    
    window_size = k_max
    start = p * p + 1
    # Initialize remainders
    # We only need to store the remainder after dividing by primes <= p
    # If any remainder > 1, it's not smooth.
    
    # Optimization: If p is very small, just use individual checks?
    # Sieve cost is O(pi(p) * window_size).
    # For p=10^5, pi(p)=9592, cost ~ 2M ops.
    # Individual factorization: 200 numbers, each trial divided by 9592 primes.
    # Cost ~ 200 * 9592 = 1.9M ops. Similar.
    # But sieve is better for finding the exact k where it breaks.
    
    # Actually, for p > 10^6, pi(p) is ~ 80k. Cost ~ 16M ops per prime p.
    # If we check 1000 primes, 16B ops. Too slow.
    # So use sieve only for p <= 10^6.
    
    if p > 1000000:
        return -1 # Signal to use individual method
    
    # Create array of values
    vals = [0] * window_size
    for i in range(window_size):
        vals[i] = start + i
        
    # Sieve with primes q <= p
    # We need primes up to p
    # _PRIMES is up to 10^7.
    
    for q in _PRIMES:
        if q > p:
            break
        # Find first multiple of q in [start, start+window_size)
        # start = p^2 + 1
        # We need i such that (p^2 + 1 + i) % q == 0
        # i % q == (-p^2 - 1) % q
        # i = (-p^2 - 1) % q
        # But we need i in [0, window_size)
        
        target = (- (p * p) - 1) % q
        # The multiples are at i = target, target + q, target + 2q, ...
        # Wait, p^2 + 1 + i.
        # (p^2 + 1 + i) % q == 0
        # i % q == (-p^2 - 1) % q
        
        # Calculate first offset
        offset = (- (p * p) - 1) % q
        
        # Iterate through window
        # This loop is inside the q loop.
        # Total operations: sum_{q<=p} (window_size / q) ~ window_size * log log p
        # Wait, the number of multiples in the window is window_size / q.
        # So total work is O(window_size * log log p).
        # This is much better than O(pi(p) * window_size).
        
        for j in range(offset, window_size, q):
            if vals[j] % q == 0:
                while vals[j] % q == 0:
                    vals[j] //= q
    
    # Check for any val > 1
    k = 0
    for i in range(window_size):
        if vals[i] == 1:
            k = i + 1
        else:
            break
            
    return k

def construct():
    deadline = time.time() + 140 # Safety margin for 150s limit
    best_k = 0
    best_p = 0
    
    # Fallback valid result
    # p=7, k=1
    best_k = 1
    best_p = 7
    
    # Iterate through primes
    # Use sieve method for p <= 10^6
    # Use individual method for p > 10^6
    
    for p in _PRIMES:
        if time.time() > deadline:
            break
        if p > 10**9:
            break
            
        # Quick check: p^2+1 must not be prime (or prime > p)
        # If p^2+1 is prime, k=0.
        # We can check this fast.
        
        if p <= 1000000:
            k = check_window_smooth_sieve(p, 200)
        else:
            # Individual checks
            k = 0
            # Check i=1
            n = p * p + 1
            if not is_p_smooth_individual(n, p):
                continue
            k = 1
            # Check i=2..200
            for i in range(2, 201):
                n = p * p + i
                if not is_p_smooth_individual(n, p):
                    break
                k = i
                
        if k > best_k:
            best_k = k
            best_p = p
            
        if best_k >= 200:
            break
            
    # If we have time, check some larger primes individually
    # But the sieve method is limited to p <= 10^6.
    # For p > 10^6, individual checks are slow.
    # We might not find better than what's in _PRIMES.
    
    return {'p': best_p, 'k': best_k}
