import math
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145  # 5s margin for safety

    # Deterministic Miller-Rabin bases for n < 2^64
    MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0:
            return False
        if n % 3 == 0:
            return n == 3
        
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
            
        for a in MR_BASES:
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

    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
            
        while True:
            x = random.randint(2, n - 1)
            y = x
            c = random.randint(1, n - 1)
            d = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d

    def factorize(n):
        factors = []
        # Trial division by very small primes first
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                while n % p == 0:
                    factors.append(p)
                    n //= p
        if n == 1:
            return factors
        if is_prime(n):
            factors.append(n)
            return factors
        
        # Pollard Rho for remaining
        stack = [n]
        while stack:
            x = stack.pop()
            if is_prime(x):
                factors.append(x)
            else:
                d = pollard_rho(x)
                stack.append(d)
                stack.append(x // d)
        return factors

    def check_run(p):
        """
        Returns the length of the run of p-smooth numbers starting from p^2+1.
        Max 200.
        """
        k = 0
        for offset in range(1, 201):
            n = p * p + offset
            # Quick check: if n is prime, it has a factor > p (itself)
            if is_prime(n):
                break
            
            # Check if all factors <= p
            factors = factorize(n)
            if max(factors) > p:
                break
            k += 1
        return k

    best_k = 0
    best_p = 2

    # Phase 1: Small primes sequential
    # Check primes up to a certain limit sequentially to find small k values
    LIMIT_SMALL = 100000
    # Generate primes up to LIMIT_SMALL using a sieve
    sieve = bytearray(b'\x01') * (LIMIT_SMALL + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(LIMIT_SMALL ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    small_primes = [i for i, is_p in enumerate(sieve) if is_p]

    for p in small_primes:
        if time.time() - start_time > DEADLINE - 30:
            break
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Random search for larger primes
    # We want to find primes p such that p^2+i is composite for many i.
    # This is equivalent to finding p such that p^2+i has a small factor.
    # We can use a random search with a filter.
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 5:
        attempts = 0
        while time.time() - start_time < DEADLINE:
            # Pick a random prime in a larger range, e.g., [10^5, 10^6]
            # We don't have a sieve for this range, so we'll just pick random odd numbers and test for primality
            # Or we can extend the sieve, but that might be too slow.
            # Let's pick a random number in [10^5, 10^6] and check if it's prime.
            p = random.randint(10**5, 10**6)
            if p % 2 == 0:
                p += 1
            if not is_prime(p):
                continue
            
            # Filter: p^2+1 must be composite
            n = p*p + 1
            if is_prime(n):
                continue
            
            k = check_run(p)
            if k > best_k:
                best_k = k
                best_p = p
            attempts += 1
            if attempts > 5000:
                break

    if best_k < 1:
        best_p = 7
        best_k = 1
        
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
