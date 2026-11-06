import time
import math
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds deadline

    # Sieve of Eratosthenes for small primes
    def sieve(limit):
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i:limit+1:i] = bytearray(len(range(i*i, limit+1, i)))
        return [i for i, prime in enumerate(is_prime) if prime]

    small_primes = sieve(100000)
    small_primes_set = set(small_primes)

    # Miller-Rabin primality test
    def is_prime_mr(n):
        if n < 2:
            return False
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n == p:
                return True
            if n % p == 0:
                return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    # Pollard's Rho (Brent variant)
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(1, n - 1)
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

    def factorize(n):
        if n == 1:
            return []
        if n in small_primes_set:
            return [n]
        factors = []
        for p in small_primes:
            if p * p > n:
                break
            if n % p == 0:
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
            if is_prime_mr(x):
                factors.append(x)
            else:
                d = pollard_brent(x)
                stack.append(d)
                stack.append(x // d)
        return factors

    def largest_prime_factor(n):
        if n == 1:
            return 1
        max_f = 1
        temp_n = n
        for p in small_primes:
            if p * p > temp_n:
                break
            if temp_n % p == 0:
                max_f = max(max_f, p)
                while temp_n % p == 0:
                    temp_n //= p
        if temp_n > 1:
            if is_prime_mr(temp_n):
                max_f = max(max_f, temp_n)
            else:
                f = factorize(temp_n)
                max_f = max(max_f, max(f))
        return max_f

    def check_window_full(p, p_sq, window_size):
        """Check if all numbers p^2+1 ... p^2+window_size have all prime factors <= p."""
        k = 0
        for j in range(1, window_size + 1):
            n = p_sq + j
            if largest_prime_factor(n) <= p:
                k = j
            else:
                break
        return k

    def batched_gcd_prefilter(p, p_sq, window_size, primes_up_to_p):
        """
        Batched GCD check: compute GCD of product(p^2+1, ..., p^2+window_size) with
        the product of all primes in (p, upper_bound]. If GCD > 1, some number has
        a prime factor > p.
        
        Returns True if all numbers potentially pass (GCD == 1), False otherwise.
        """
        # We need to check primes in range (p, p^2 + window_size]
        # But we can't generate all such primes. Instead, use a segmented approach.
        # 
        # Alternative: For each prime q > p that could divide some p^2+j (1<=j<=window_size),
        # we have p^2+j ≡ 0 (mod q), so j ≡ -p^2 (mod q).
        # Since 1 <= j <= window_size, we need -p^2 mod q to be in [1, window_size].
        # 
        # Key insight: if q > p^2 + window_size, then q cannot divide any p^2+j.
        # If p < q <= p^2 + window_size, then q divides p^2+j iff j ≡ (-p^2) mod q.
        # Since q > p, (-p^2) mod q = q - (p^2 mod q) if p^2 mod q != 0, else 0.
        # But p^2 mod q: since q > p, p^2 < q^2, so p^2 mod q = p^2 if p^2 < q, else p^2 - q.
        # Actually p^2 mod q is just p^2 % q.
        # 
        # For the batched approach, let's compute the product P = prod(p^2+j for j in 1..window_size)
        # and check gcd(P, M) where M is product of primes in (p, some_limit].
        # But M is enormous. Instead, we check gcd of P with each prime individually in a batch.
        #
        # Better: Use the fact that if any p^2+j has a prime factor > p, then that prime
        # must be in (p, p^2+j]. We can check by computing for each j whether p^2+j 
        # has a factor > p using a segmented sieve approach.
        #
        # Segmented sieve approach:
        # For primes q in (p, p^2 + window_size], mark positions where q divides p^2+j.
        # If any position 1..window_size is marked, that number has a factor > p.
        
        # Generate primes in range (p, p^2 + window_size] using segmented sieve
        # But this range can be huge. Instead, note that for q > p^2 + window_size,
        # q cannot divide any p^2+j. And for q in (sqrt(p^2+window_size), p^2+window_size],
        # q can divide at most one number in the window.
        
        # Practical approach: Check each number individually but with early termination.
        # Use trial division by primes up to min(p, 100000), then check remainder.
        
        max_trial = min(p, 100000)
        trial_primes = [q for q in small_primes if q <= max_trial]
        
        for j in range(1, window_size + 1):
            n = p_sq + j
            temp = n
            for q in trial_primes:
                if q * q > temp:
                    break
                while temp % q == 0:
                    temp //= q
            
            if temp == 1:
                continue
            elif temp <= p:
                # All remaining factors are <= p, so this number passes
                continue
            else:
                # temp > p, so there's a prime factor > p
                return False
        
        return True

    best_k = 0
    best_p = 2

    # Generate primes for p up to a reasonable limit
    limit_p = 10**7
    primes_p = sieve(limit_p)

    # Strategy: 
    # 1. First pass: quick filter using batched approach on many candidates
    # 2. Second pass: full verification on candidates that pass the filter
    
    for p in primes_p:
        if time.time() > deadline:
            break
        
        p_sq = p * p
        
        # Quick check: can we possibly beat best_k?
        # We need at least best_k + 1 consecutive numbers with all factors <= p
        # Quick necessary condition: p^2+1 through p^2+best_k+1 must not have
        # any prime factor > p
        
        # Use the batched prefilter for a quick check
        if not batched_gcd_prefilter(p, p_sq, best_k + 1, small_primes):
            continue
        
        # Passed prefilter, do full verification
        k = check_window_full(p, p_sq, 200)
        
        if k > best_k:
            best_k = k
            best_p = p
    
    return {'p': best_p, 'k': best_k}
