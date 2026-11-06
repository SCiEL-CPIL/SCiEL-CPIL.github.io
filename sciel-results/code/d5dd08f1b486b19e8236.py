import math
import time
import random

def construct():
    # Time limit guard
    start_time = time.time()
    TIME_LIMIT = 115  # 5 seconds safety margin
    
    # Miller-Rabin for determinism on 64-bit integers
    def is_prime(n):
        if n < 2:
            return False
        # Small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n % p == 0:
                return n == p
        # Write n-1 as d*2^s
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        # Bases sufficient for n < 2^64
        # See https://miller-rabin.appspot.com/
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

    # Pollard's Brent's cycle detection
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        
        y, c, m = random.randrange(1, n), random.randrange(1, n), random.randrange(1, n)
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
        return g

    def has_large_factor(n, p):
        """
        Returns True if n has any prime factor > p.
        Returns False if all prime factors are <= p.
        """
        while n > 1:
            if is_prime(n):
                return n > p
            
            # Find a factor
            # Simple trial division for small factors to speed up Pollard
            # Trial division up to 100 is cheap
            f = None
            for d in range(2, 100):
                if n % d == 0:
                    f = d
                    break
            
            if f is None:
                f = pollard_brent(n)
            
            if f > p:
                return True
            n //= f
            # Also check if the cofactor is prime and > p
            # The loop will handle it in the next iteration or is_prime check
        return False

    def generate_small_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, prime in enumerate(sieve) if prime]

    def get_random_prime(min_val, max_val):
        """Generate a random prime in [min_val, max_val]"""
        while True:
            n = random.randint(min_val, max_val)
            if n % 2 == 0:
                n += 1
            if is_prime(n):
                return n

    best_p = 0
    best_k = 0
    
    # Phase 1: Quick scan of small primes to establish a baseline
    small_primes = generate_small_primes(10**5)
    for p in small_primes:
        if time.time() - start_time > TIME_LIMIT:
            break
        k = 0
        base = p * p
        while k < 200:
            if time.time() - start_time > TIME_LIMIT:
                break
            n = base + k + 1
            if has_large_factor(n, p):
                break
            k += 1
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Random search in larger ranges
    # Larger p are more likely to have longer runs of p-smooth numbers
    ranges = [
        (10**5, 10**6),
        (10**6, 10**7),
        (10**7, 10**8),
        (10**8, 10**9)
    ]
    
    for min_val, max_val in ranges:
        if time.time() - start_time > TIME_LIMIT:
            break
        
        # Try a number of random primes in this range
        num_attempts = 50
        for _ in range(num_attempts):
            if time.time() - start_time > TIME_LIMIT:
                break
            
            p = get_random_prime(min_val, max_val)
            k = 0
            base = p * p
            while k < 200:
                if time.time() - start_time > TIME_LIMIT:
                    break
                n = base + k + 1
                if has_large_factor(n, p):
                    break
                k += 1
            
            if k > best_k:
                best_k = k
                best_p = p

    # Ensure we have a valid result
    if best_p == 0 or best_k == 0:
        # Fallback: find the first valid small prime
        for p in small_primes:
            k = 0
            base = p * p
            while k < 200:
                n = base + k + 1
                if has_large_factor(n, p):
                    break
                k += 1
            if k >= 1:
                best_p = p
                best_k = k
                break
    
    return {'p': best_p, 'k': best_k}

if __name__ == '__main__':
    construct()
