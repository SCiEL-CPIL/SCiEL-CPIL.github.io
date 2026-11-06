import time
import random
import math

def is_prime_mr(n):
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    # Deterministic witnesses for n < 3,317,044,064,679,887,385,961,981
    # We only need up to 10^9 or so, so first few witnesses suffice
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for a in witnesses:
        if a >= n:
            continue
        x = pow(a, n - 1, n)
        if x == 1 or x == n - 1:
            continue
        composite = True
        for _ in range(math.floor(math.log2(n - 1))):
            x = (x * x) % n
            if x == n - 1:
                composite = False
                break
        if composite:
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
        m = random.randrange(1, n)
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
    if n == 1:
        return True
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
    if n > p:
        # If n is prime, it's a factor > p (since n > p)
        # If n is composite, it might have factors <= p
        # But if n > p and we haven't divided by anything > p yet...
        # Actually, if n > p, it could be a prime factor > p, or a product of factors <= p.
        # We need to check if n has ANY prime factor > p.
        # If n is prime, return n <= p (which is False since n > p).
        if is_prime_mr(n):
            return n <= p
        # If composite, find a factor
        # Use Pollard Brent
        # But we must ensure we don't miss that a factor might be > p
        # Let's just factor n completely and check
        # For efficiency, we can just find one factor and recurse
        # But recursion depth might be an issue? n is small enough.
        # Let's use a helper to check if all factors of n are <= p
        def all_factors_leq(n, p):
            if n == 1:
                return True
            if is_prime_mr(n):
                return n <= p
            q = pollard_brent(n)
            return all_factors_leq(q, p) and all_factors_leq(n // q, p)
        return all_factors_leq(n, p)
    else:
        # n <= p
        # If n is prime, it's <= p, so OK.
        # If composite, factors are < n <= p, so OK.
        return True

def sieve(limit):
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.sqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s margin
    
    # Generate small primes for trial division
    small_primes = sieve(10000)
    
    # Generate candidate primes p
    # We'll check primes up to 10^6 first
    candidate_limit = 1000000
    primes = sieve(candidate_limit)
    
    best_k = 0
    best_p = 2
    
    for p in primes:
        if time.time() > deadline:
            break
        # Check p^2 + 1
        n = p * p + 1
        if not is_p_smooth(n, p, small_primes):
            continue
        
        k = 1
        # Check subsequent numbers
        while True:
            if time.time() > deadline:
                break
            k += 1
            if k > 200:
                break
            n = p * p + k
            if not is_p_smooth(n, p, small_primes):
                break
            if k > best_k:
                best_k = k
                best_p = p
        if k > best_k:
            best_k = k
            best_p = p
            
    return {'p': best_p, 'k': best_k}
