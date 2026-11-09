import time
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 115.0  # safety margin before 120s limit

    # Deterministic Miller-Rabin for n < 3.3e24
    def is_prime_mr(n):
        if n < 2:
            return False
        if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            return True
        if n % 2 == 0:
            return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
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

    # Pollard's Brent cycle detection
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(2, n - 1)
            c = random.randint(1, n - 1)
            m = random.randint(1, n - 1)
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

    # Trial division by small primes up to limit
    def trial_division(n, limit=10000):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        d = 5
        while d * d <= n and d <= limit:
            if n % d == 0:
                return d
            if n % (d + 2) == 0:
                return d + 2
            d += 6
        return None

    # Check if n is p-smooth (all prime factors <= p)
    def is_p_smooth(n, p):
        if n <= 1:
            return True
        if n <= p:
            return is_prime_mr(n) or True  # if n <= p, any factor is <= p
        # First check if n itself is prime
        if is_prime_mr(n):
            return n <= p
        # Trial division by small primes
        small_factor = trial_division(n, limit=1000)
        if small_factor is not None:
            if small_factor > p:
                return False
            return is_p_smooth(n // small_factor, p)
        # Use Pollard-Brent
        factor = pollard_brent(n)
        if factor > p:
            return False
        return is_p_smooth(n // factor, p) and is_p_smooth(factor, p)

    # Better: check p-smoothness by full factorization
    def factorize(n):
        """Return list of prime factors of n."""
        if n <= 1:
            return []
        factors = []
        # Small primes trial division
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for sp in small_primes:
            while n % sp == 0:
                factors.append(sp)
                n //= sp
        if n == 1:
            return factors
        if is_prime_mr(n):
            factors.append(n)
            return factors
        # Pollard-Brent for remaining
        factor = pollard_brent(n)
        factors.extend(factorize(factor))
        factors.extend(factorize(n // factor))
        return factors

    def is_p_smooth_full(n, p):
        """Check if all prime factors of n are <= p."""
        if n <= 1:
            return True
        if n <= p:
            return True
        factors = factorize(n)
        return all(f <= p for f in factors)

    # Segmented sieve for interval [p^2, p^2 + k_max]
    # We want to check if each number in [p^2+1, p^2+k] has all prime factors <= p
    # Strategy: For each prime q <= p, mark multiples of q in the interval
    # A number is p-smooth iff it is divisible by some prime <= p (or equals 1)
    # Actually, we need ALL prime factors <= p, which means the number must be
    # completely factored into primes <= p.
    
    # Better approach: For the interval [p^2+1, p^2+k], check p-smoothness
    # by dividing out all small primes <= p and seeing if remainder is 1.
    
    # Precompute primes up to some limit for trial division
    def sieve_primes(limit):
        is_p = [True] * (limit + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                for j in range(i*i, limit+1, i):
                    is_p[j] = False
        return [i for i in range(2, limit+1) if is_p[i]]

    # For a given p, check the run length
    def check_run(p, k_max=200):
        """Check how many consecutive integers p^2+1, p^2+2, ... are p-smooth."""
        base = p * p
        # For each number in [base+1, base+k_max], check if it's p-smooth
        # Efficient: for each number, try to divide by primes <= p
        # Since p can be up to 10^9, we can't enumerate all primes <= p
        # Instead: a number n is p-smooth iff after dividing by all primes <= p, remainder is 1
        # But we can't enumerate all primes <= p if p is large.
        
        # Alternative: factorize each number and check max factor <= p
        # For numbers near p^2, they have at most ~2*log2(p^2) = 4*log2(p) bits
        # So factorization should be feasible with Pollard-Brent
        
        k = 0
        for i in range(1, k_max + 1):
            n = base + i
            if is_p_smooth_full(n, p):
                k = i
            else:
                break
        return k

    # Search over primes p
    # Start with known good primes and search systematically
    best_k = 0
    best_p = 2
    
    # Generate candidate primes to test
    # First, test small primes exhaustively
    # Then test larger primes
    
    # Precompute primes up to a reasonable limit for quick testing
    small_primes = sieve_primes(100000)
    
    # Test all small primes
    for p in small_primes:
        if time.time() - start_time > DEADLINE:
            break
        k = check_run(p, 200)
        if k > best_k:
            best_k = k
            best_p = p
        if best_k >= 200:
            break
    
    # If we haven't found k=200, try larger primes
    # Use a primality test to find primes in larger ranges
    if best_k < 200 and time.time() - start_time < DEADLINE:
        # Try primes in ranges, sampling
        test_ranges = [
            (100000, 500000),
            (500000, 2000000),
            (2000000, 10000000),
            (10000000, 50000000),
            (50000000, 200000000),
            (200000000, 1000000000),
        ]
        
        for lo, hi in test_ranges:
            if time.time() - start_time > DEADLINE:
                break
            # Sample primes in this range
            # Use a step to avoid testing every odd number
            step = max(1, (hi - lo) // 10000)  # test at most ~10000 candidates per range
            for p in range(lo, hi, step):
                if time.time() - start_time > DEADLINE:
                    break
                if p < 2:
                    continue
                # Quick primality check
                if p % 2 == 0:
                    continue
                if is_prime_mr(p):
                    k = check_run(p, 200)
                    if k > best_k:
                        best_k = k
                        best_p = p
                        if best_k >= 200:
                            break
            if best_k >= 200:
                break
    
    return {'p': best_p, 'k': best_k}
