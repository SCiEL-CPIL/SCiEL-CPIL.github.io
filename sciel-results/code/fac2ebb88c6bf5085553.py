import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 118.0  # Safety margin for 150s limit
    
    # Deterministic Miller-Rabin for 64-bit integers
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return n == 2 or n == 3
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
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True
    
    # Precompute small primes up to 100000 for trial division
    def sieve(n):
        bs = [True] * (n + 1)
        bs[0] = bs[1] = False
        for i in range(2, int(n**0.5) + 1):
            if bs[i]:
                for j in range(i*i, n+1, i):
                    bs[j] = False
        return [i for i, b in enumerate(bs) if b]
    
    small_primes = sieve(100000)
    
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
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
    
    def largest_prime_factor(n):
        """Find the largest prime factor of n."""
        if n <= 1:
            return 1
        if is_prime(n):
            return n
        # Trial division by small primes
        for sp in small_primes:
            if sp * sp > n:
                break
            if n % sp == 0:
                while n % sp == 0:
                    n //= sp
                if n == 1:
                    return sp
                if is_prime(n):
                    return n
                # Continue with reduced n
                break
        # If we broke out, n is either 1, prime, or composite with no small factors
        if n == 1:
            # The largest factor was the last small prime we divided by
            # Actually we need to track this. Let me restructure.
            pass
        
        # Better approach: fully factor using Pollard
        factors = []
        while n > 1:
            if is_prime(n):
                factors.append(n)
                break
            d = pollard_brent(n)
            if d == n:
                continue
            factors.append(d)
            n //= d
        return max(factors)
    
    def is_p_smooth(n, p):
        """Check if all prime factors of n are <= p."""
        if n <= 1:
            return True
        if is_prime(n):
            return n <= p
        # Trial divide by small primes <= p
        for sp in small_primes:
            if sp > p:
                break
            if n % sp == 0:
                while n % sp == 0:
                    n //= sp
                if n == 1:
                    return True
        if n == 1:
            return True
        # n is now composed of primes > 100000 (or > p if p < 100000)
        # If n is prime, check if <= p
        if is_prime(n):
            return n <= p
        # n is composite. Need to check if it has any factor > p
        # Find largest prime factor
        lpf = largest_prime_factor(n)
        return lpf <= p
    
    best_k = 0
    best_p = 2
    
    # Check small primes exhaustively first
    for p in range(2, 10000, 1):
        if not is_prime(p):
            continue
        p2 = p * p
        k = 0
        while k < 200:
            val = p2 + k + 1
            if not is_p_smooth(val, p):
                break
            k += 1
        if k > best_k:
            best_k = k
            best_p = p
    
    # Now search larger primes
    # Focus on ranges where runs might be longer
    # Use a combination of random and structured search
    search_ranges = [
        (10000, 100000),
        (100000, 1000000),
        (1000000, 10000000),
        (10000000, 100000000),
        (100000000, 1000000000),
    ]
    
    # Try to find primes in each range
    for lo, hi in search_ranges:
        if time.time() - start_time > DEADLINE:
            break
        # Sample some random primes in this range
        attempts = 0
        while time.time() - start_time < DEADLINE and attempts < 500:
            attempts += 1
            p = random.randint(lo, hi)
            p |= 1  # Make odd
            if p < 2:
                continue
            if not is_prime(p):
                continue
            
            p2 = p * p
            k = 0
            while k < 200:
                val = p2 + k + 1
                if not is_p_smooth(val, p):
                    break
                k += 1
            if k > best_k:
                best_k = k
                best_p = p
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
