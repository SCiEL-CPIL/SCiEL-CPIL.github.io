import time
import math
import random

def construct():
    start_time = time.time()
    deadline = 115  # seconds, safety margin
    
    # Deterministic Miller-Rabin for n < 3.3 * 10^24
    # Witnesses: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
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
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in small_primes:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
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
        while True:
            y = random.randint(1, n - 1)
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
                        q = (q * abs(x - y)) % n
                    g = math.gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                while True:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
                    if g != 1:
                        break
            if g != n:
                return g

    # Pre-compute small primes for trial division
    def sieve_small(limit):
        is_p = bytearray([1]) * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit ** 0.5) + 1):
            if is_p[i]:
                for j in range(i * i, limit + 1, i):
                    is_p[j] = 0
        return [i for i in range(limit + 1) if is_p[i]]

    small_primes = sieve_small(100000) # Primes up to 100k

    def is_p_smooth(n, p):
        if n <= p:
            return True
        if is_prime(n):
            return n <= p
        
        # Trial division by small primes
        for q in small_primes:
            if q > p:
                break
            while n % q == 0:
                n //= q
                if n <= p:
                    return True
        
        if n == 1:
            return True
        if n <= p:
            return True
        
        if is_prime(n):
            return False
        
        d = pollard_rho(n)
        if d == n:
            return False
        
        return is_p_smooth(d, p) and is_p_smooth(n // d, p)

    # Segmented Sieve for primes up to 10^9
    def segmented_sieve(limit):
        # Base primes up to sqrt(limit)
        sqrt_limit = int(math.isqrt(limit))
        base_primes = sieve_small(sqrt_limit)
        
        # Process segments
        seg_size = 10**6
        low = 2
        while low <= limit:
            high = min(low + seg_size - 1, limit)
            is_prime_seg = bytearray([1]) * (high - low + 1)
            
            for p in base_primes:
                if p * p > high:
                    break
                # Find first multiple of p in [low, high]
                start = ((low + p - 1) // p) * p
                if start == p:
                    start += p
                for j in range(start, high + 1, p):
                    is_prime_seg[j - low] = 0
            
            for i in range(high - low + 1):
                if is_prime_seg[i]:
                    yield low + i
            
            low = high + 1

    best_k = 0
    best_p = 2
    
    # Check small primes first
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        k = 0
        while k < 200:
            n = p * p + k + 1
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p

    # Search larger primes
    # We search up to 10^9
    # But we need to be careful with time
    # Let's iterate through primes from segmented sieve
    # We skip primes already checked (up to 100)
    
    count = 0
    for p in segmented_sieve(10**9):
        if p <= 100:
            continue
        if time.time() - start_time > deadline - 5:
            break
        
        k = 0
        while k < 200:
            n = p * p + k + 1
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
            
        count += 1
        if count % 10000 == 0:
            if time.time() - start_time > deadline - 10:
                break

    return {'p': best_p, 'k': best_k}

if __name__ == "__main__":
    res = construct()
    print(res)
