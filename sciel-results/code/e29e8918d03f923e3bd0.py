"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random

def is_prime_mr(n):
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
    # Deterministic bases for n < 2^64
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def sieve_primes(limit):
    if limit < 2:
        return []
    is_p = bytearray([1]) * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_p[i]:
            is_p[i*i : limit+1 : i] = bytearray(len(is_p[i*i : limit+1 : i]))
    return [i for i, v in enumerate(is_p) if v]

def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    y, c, m = random.randint(1, n - 1), random.randint(1, n - 1), random.randint(1, n - 1)
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
    return g

def is_p_smooth(n, p, small_primes):
    """Check if n is p-smooth using trial division by small primes and Pollard-Brent for the rest."""
    temp_n = n
    # 1. Trial divide by small primes (precomputed)
    for q in small_primes:
        if q > p:
            break
        if temp_n % q == 0:
            while temp_n % q == 0:
                temp_n //= q
            if temp_n == 1:
                return True
        # Optimization: if q*q > temp_n, temp_n is prime
        if q * q > temp_n:
            break
    
    if temp_n == 1:
        return True
    
    # 2. Check if remaining temp_n is prime
    if is_prime_mr(temp_n):
        return temp_n <= p
    
    # 3. Use Pollard-Brent to find a factor
    # We need to ensure all factors are <= p
    # If we find a factor d > p, it fails.
    # If d <= p, we divide and check the rest.
    
    stack = [temp_n]
    while stack:
        x = stack.pop()
        if x == 1:
            continue
        if is_prime_mr(x):
            if x > p:
                return False
            continue
        
        # Find a factor
        d = pollard_brent(x)
        if d == x:
            # Failed to find a factor, might be prime? MR said no, but maybe MR error or cycle?
            # Retry with different params? Or just assume failure?
            # For robustness, if pollard fails, we might need to fallback or retry.
            # Let's retry a few times.
            found = False
            for _ in range(5):
                d = pollard_brent(x)
                if d != x:
                    found = True
                    break
            if not found:
                # If we can't factor it, and it's not prime, it's likely composite with large factors.
                # If x > p, and it's composite, it has a prime factor <= sqrt(x).
                # If sqrt(x) > p, it could still be p-smooth? No, if x is p-smooth, all factors <= p.
                # So if x is composite and p-smooth, it has a factor <= p.
                # If pollard fails, it's a bug or extreme case.
                # Let's assume failure to be safe? Or try trial division up to p?
                # Given time constraints, assume failure.
                return False
        
        # We found a factor d
        # Check if d > p
        if d > p:
            return False
        
        # Divide x by d
        while x % d == 0:
            x //= d
        
        if x > 1:
            stack.append(x)
            
    return True

def construct():
    # Search for p up to a reasonable limit.
    # With faster factorization, we can check more p's or larger p's.
    # Let's aim for p up to 10^7 or 10^8?
    # Sieve memory for 10^8 is 100MB, which is okay. Time for sieve is ~1s.
    # But we only need small primes for trial division (e.g., up to 10^5).
    # We need to iterate p. We can generate primes on the fly or sieve up to a limit.
    
    LIMIT_P = 10**6 # Start with 10^6, maybe increase if fast
    small_prime_limit = 10**5
    
    small_primes = sieve_primes(small_prime_limit)
    primes = sieve_primes(LIMIT_P)
    
    best_k = 0
    best_p = 2
    
    for p in primes:
        if p > LIMIT_P:
            break
        
        k = 0
        for i in range(1, 201):
            n = p * p + i
            
            if is_p_smooth(n, p, small_primes):
                k += 1
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p
            
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
