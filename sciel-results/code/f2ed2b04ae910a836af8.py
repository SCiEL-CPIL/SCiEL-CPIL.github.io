import time
import math
import numpy as np
import random

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin
    
    # 1. Sieve primes up to B = 10^6
    B = 10**6
    sieve = np.ones(B + 1, dtype=bool)
    sieve[0:2] = False
    for i in range(2, int(math.isqrt(B)) + 1):
        if sieve[i]:
            sieve[i*i:B+1:i] = False
    primes = np.array(np.nonzero(sieve))[0]
    primes_list = primes.tolist()
    
    # 2. Miller-Rabin for primality test
    def is_prime(n):
        if n < 2:
            return False
        if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            return True
        if n % 2 == 0 or n % 3 == 0 or n % 5 == 0 or n % 7 == 0 or n % 11 == 0 or n % 13 == 0 or n % 17 == 0 or n % 19 == 0 or n % 23 == 0 or n % 29 == 0 or n % 31 == 0 or n % 37 == 0:
            return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in [2, 325, 9375, 28178, 450775, 9780504, 1795265022]:
            if a % n == 0:
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

    # 3. Pollard-Brent
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(2, n - 1)
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
            if g != n:
                return g

    # 4. Smoothness Check
    def is_p_smooth(n, p):
        if n <= 1:
            return True
        if p < 2:
            return n == 1
            
        if p <= B:
            # Fast path: divide by all primes <= p
            idx = np.searchsorted(primes, p, side='right')
            for i in range(idx):
                q = primes_list[i]
                if q * q > n:
                    break
                while n % q == 0:
                    n //= q
                if n == 1:
                    return True
            return n == 1
        else:
            # Slow path: divide by primes <= B, then factor remainder
            idx = len(primes_list)
            for i in range(idx):
                q = primes_list[i]
                if q * q > n:
                    break
                while n % q == 0:
                    n //= q
                if n == 1:
                    return True
                if time.time() - start_time > timeout:
                    return False
            
            if n == 1:
                return True
            
            # n has no factors <= B.
            # Check if n is p-smooth.
            # Use stack to factor n into primes.
            stack = [n]
            while stack:
                if time.time() - start_time > timeout:
                    return False
                m = stack.pop()
                if m == 1:
                    continue
                if is_prime(m):
                    if m > p:
                        return False
                else:
                    f = pollard_brent(m)
                    if f == m:
                        # Failed to factor, assume prime? Or retry?
                        # If pollard_brent returns m, it failed.
                        # Retry with different params?
                        # For now, treat as prime for safety? No, if it's composite with factor > p, we fail.
                        # If it's composite with factors <= p, we pass.
                        # This is a risk.
                        # Let's just return False to be safe? Or try again?
                        # Given time constraints, return False.
                        return False
                    stack.append(f)
                    stack.append(m // f)
            return True

    best_k = 0
    best_p = 2

    # Phase 1: Search p <= B
    for p in primes_list:
        if time.time() - start_time > timeout:
            break
        k = 0
        p_sq = p * p
        while k < 200:
            n = p_sq + k + 1
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
            if best_k == 200:
                break

    # Phase 2: Search p > B (if time remains)
    # Iterate odd numbers, check primality, then smoothness.
    # To be efficient, skip multiples of small primes.
    # We can use a wheel or just check is_prime.
    # Since B=10^6, we can go up to ~10^7 or 10^8 depending on speed.
    next_p = B + 1
    if next_p % 2 == 0:
        next_p += 1
        
    # We will check candidates.
    # To speed up, we can check a few candidates at a time?
    # No, just iterate.
    
    # Optimization: Only check p if p^2+1 is smooth.
    # This is a strong filter.
    
    while next_p < 10**8:
        if time.time() - start_time > timeout:
            break
        
        if is_prime(next_p):
            k = 0
            p_sq = next_p * next_p
            while k < 200:
                n = p_sq + k + 1
                if is_p_smooth(n, next_p):
                    k += 1
                else:
                    break
            if k > best_k:
                best_k = k
                best_p = next_p
                if best_k == 200:
                    break
        next_p += 2

    return {'p': best_p, 'k': best_k}
