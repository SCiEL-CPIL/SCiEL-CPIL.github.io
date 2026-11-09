import time
from math import isqrt, gcd

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin before 150s kill
    
    # --- Number Theory Utilities ---
    
    def is_prime_mr(n):
        if n < 2:
            return False
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
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

    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        y, c, m = 2, 1, 128
        g, q, r, x = 1, 1, 1, 0
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
                g = gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = gcd(abs(x - ys), n)
                if g > 1:
                    break
        return g

    def factorize(n):
        factors = []
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            while n % p == 0:
                factors.append(p)
                n //= p
        if n == 1:
            return factors
        if is_prime_mr(n):
            factors.append(n)
            return factors
        stack = [n]
        while stack:
            x = stack.pop()
            if x == 1:
                continue
            if is_prime_mr(x):
                factors.append(x)
                continue
            d = pollard_brent(x)
            stack.append(d)
            stack.append(x // d)
        return factors

    def is_p_smooth(n, p, small_primes):
        # Trial divide by small primes
        limit = min(p, isqrt(n))
        for q in small_primes:
            if q > limit:
                break
            while n % q == 0:
                n //= q
                if n == 1:
                    return True
        
        if n == 1:
            return True
        
        # If n <= p, all remaining factors are <= n <= p
        if n <= p:
            return True
        
        # If n > p, we need to check if n has any factor <= p
        # But we already removed all factors <= limit.
        # If limit = p, then n has no factors <= p.
        # Since n > p, and no factors <= p, n is not p-smooth.
        if limit == p:
            return False
        
        # If limit < p (i.e., isqrt(n) < p), then n is prime.
        # Since n > p, it is not p-smooth.
        # Wait, if isqrt(n) < p, then n is prime.
        # If n is prime and n > p, then False.
        return False

    # --- Main Search ---
    
    # Sieve small primes
    SIEVE_LIMIT = 10**6
    sieve = bytearray(b'\x01') * (SIEVE_LIMIT + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(SIEVE_LIMIT) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    small_primes = [i for i in range(2, SIEVE_LIMIT + 1) if sieve[i]]
    
    best_k = 0
    best_p = 2
    
    p = 2
    while True:
        if time.time() > DEADLINE:
            break
        
        # Check if p is prime
        if p <= SIEVE_LIMIT:
            if not sieve[p]:
                p += 1 if p == 2 else 2
                continue
        else:
            if not is_prime_mr(p):
                p += 1 if p == 2 else 2
                continue
        
        # p is prime. Check run length
        k = 0
        while k < 200:
            if time.time() > DEADLINE:
                break
            n = p * p + k + 1
            if is_p_smooth(n, p, small_primes):
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
            
        # Next p
        if p == 2:
            p = 3
        else:
            p += 2

    return {'p': best_p, 'k': best_k}
