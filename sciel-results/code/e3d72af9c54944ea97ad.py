import time
import math
import random

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    # Small primes for trial division
    def sieve(limit):
        is_prime = [True] * (limit + 1)
        is_prime[0] = is_prime[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                for j in range(i*i, limit + 1, i):
                    is_prime[j] = False
        return [i for i, prime in enumerate(is_prime) if prime]
    
    small_primes = sieve(10000)
    
    # Miller-Rabin primality test
    def is_prime(n):
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0:
            return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
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

    # Pollard's Rho with Brent's improvement
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        y, c, m = random.randint(1, n - 1), random.randint(1, n - 1), 128
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

    def factorize(n):
        """Returns list of prime factors of n."""
        if n == 1:
            return []
        for p in small_primes:
            if p * p > n:
                break
            if n % p == 0:
                return [p] + factorize(n // p)
        if is_prime(n):
            return [n]
        d = pollard_brent(n)
        return factorize(d) + factorize(n // d)

    def largest_prime_factor(n):
        """Returns the largest prime factor of n."""
        if n == 1:
            return 1
        max_p = 1
        for p in small_primes:
            if p * p > n:
                break
            if n % p == 0:
                max_p = max(max_p, p)
                while n % p == 0:
                    n //= p
        if n == 1:
            return max_p
        if is_prime(n):
            return max(max_p, n)
        # If n is composite, factorize it
        # We can just get all factors and take max
        factors = factorize(n)
        return max(max_p, max(factors))

    def is_p_smooth(n, p):
        """Checks if all prime factors of n are <= p."""
        if n == 1:
            return True
        # Quick check: if n is prime
        if is_prime(n):
            return n <= p
        # Try small primes
        temp_n = n
        for q in small_primes:
            if q > p:
                # If we encounter a small prime q > p, and it divides n, then not smooth
                # But wait, if q > p, we don't need to check further for smoothness?
                # No, if q divides n and q > p, then n is not p-smooth.
                # But we only iterate small primes up to 10000.
                # If p < 10000, we might miss a factor.
                # Let's just use largest_prime_factor for simplicity and correctness, 
                # but optimize by early exit if possible.
                pass
            if q * q > temp_n:
                break
            if temp_n % q == 0:
                if q > p:
                    return False
                while temp_n % q == 0:
                    temp_n //= q
        if temp_n == 1:
            return True
        if is_prime(temp_n):
            return temp_n <= p
        # temp_n is composite
        # Find a factor
        d = pollard_brent(temp_n)
        # Check d and temp_n//d
        return is_p_smooth(d, p) and is_p_smooth(temp_n // d, p)

    # Generate primes up to some limit
    # Let's try up to 1,000,000 first, then extend if time permits
    limit = 1000000
    is_prime_sieve = [True] * (limit + 1)
    is_prime_sieve[0] = is_prime_sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime_sieve[i]:
            for j in range(i*i, limit + 1, i):
                is_prime_sieve[j] = False
    
    primes = [i for i, val in enumerate(is_prime_sieve) if val]
    
    best_k = 0
    best_p = 0
    
    for p in primes:
        if time.time() - start_time > time_limit:
            break
        
        # Check p^2 + 1
        # If p^2 + 1 is not p-smooth, k=0
        # We want to find max k
        k = 0
        # Max k is 200
        for m in range(1, 201):
            if time.time() - start_time > time_limit:
                break
            val = p * p + m
            if is_p_smooth(val, p):
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
            # Optional: print progress
            # print(f"p={p}, k={k}")
            
    return {'p': best_p, 'k': best_k}
