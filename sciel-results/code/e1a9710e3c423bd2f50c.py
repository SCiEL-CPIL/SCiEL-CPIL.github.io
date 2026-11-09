import time
import random
import math

def construct():
    start_time = time.time()
    DEADLINE = 115  # seconds, safety margin
    
    def is_prime_mr(n):
        """Deterministic Miller-Rabin for n < 2^64"""
        if n < 2:
            return False
        if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            return True
        if n % 2 == 0:
            return False
        # Write n-1 as 2^r * d
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        # Witnesses sufficient for n < 3,317,044,064,679,887,385,961,981
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = x * x % n
                if x == n - 1:
                    break
            else:
                return False
        return True
    
    def sieve(n):
        """Generate all primes up to n using simple sieve"""
        if n < 2:
            return []
        is_p = bytearray(b'\x01') * (n + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(n**0.5) + 1):
            if is_p[i]:
                for j in range(i*i, n+1, i):
                    is_p[j] = 0
        return [i for i, v in enumerate(is_p) if v]
    
    # Precompute primes up to sqrt(10^9) ≈ 31623
    max_sqrt = 31623
    small_primes = sieve(max_sqrt)
    
    def segmented_spf_check(p, base, k_max=200):
        """
        For a given prime p and base=p^2, check p-smoothness of base+1, ..., base+k_max
        using segmented sieve to find smallest prime factors.
        Returns the maximum k such that all numbers base+1..base+k are p-smooth.
        """
        # We need to check numbers base+1 to base+k_max
        # A number n is p-smooth iff its largest prime factor <= p
        # Equivalently, after dividing out all prime factors <= p, we should get 1
        
        # Strategy: For each number in [base+1, base+k_max], we need to verify
        # that it has no prime factor > p.
        
        # Key insight: If n has a prime factor q > p, then q > p >= sqrt(base) for large p
        # (since base = p^2, sqrt(base) = p)
        # So any prime factor > p must be > sqrt(n) for n < p^2 + p
        # Actually for n = p^2 + j where j <= 200, sqrt(n) ≈ p + j/(2p)
        # So a prime factor > p could be as small as p+1
        
        # Better approach: Use the fact that if n is not p-smooth, it has a prime factor
        # in (p, n]. Since n < p^2 + 201, and p is large, we can check:
        # - Trial divide by all primes <= p (but p can be up to 10^9, so we can't enumerate)
        # - Instead: if n has a prime factor > p, then n = q * m where q > p and m < n/q <= (p^2+201)/(p+1) ≈ p
        #   So m < p, meaning m has all prime factors <= p (if m is p-smooth) or m itself has a factor > p
        #   This gets circular.
        
        # Practical approach for the search:
        # For each candidate number n = base + j:
        # 1. Trial divide by small primes (up to some limit like 10000)
        # 2. If remainder is 1, it's p-smooth
        # 3. If remainder is prime, check if <= p
        # 4. If remainder is composite, use Pollard-Brent
        
        # But the directive suggests segmented sieve. Let me think about this differently.
        
        # Segmented sieve approach:
        # For the interval [base+1, base+k_max], we can mark which numbers have small prime factors.
        # But we need to check if ALL prime factors are <= p, not just find small ones.
        
        # Alternative: For each n in the range, compute the largest prime factor.
        # If LPF(n) <= p, then n is p-smooth.
        
        # To compute LPF efficiently for 200 consecutive numbers starting at p^2+1:
        # We can use a segmented sieve to find all prime factors up to sqrt(max_n)
        # then check if the remaining cofactor is 1 or prime <= p.
        
        max_n = base + k_max
        # sqrt(max_n) ≈ p + small
        # We'll trial divide by primes up to min(p, sqrt(max_n))
        # But p can be large, so we need to be smart.
        
        # For each offset j from 1 to k_max:
        #   n = base + j
        #   We want to know if LPF(n) <= p
        
        # Let's use a hybrid approach:
        # 1. Pre-compute remainders of base mod small primes
        # 2. For each n, trial divide by primes <= min(p, 10000)
        # 3. Check if remaining cofactor is 1, prime <= p, or composite
        
        # Actually, let me just use the existing is_p_smooth but optimize it.
        
        # The key optimization: for consecutive numbers, we can cache some work.
        # But more importantly, let's use a better trial division limit.
        
        # For n ≈ p^2, if n has a prime factor > p, that factor is at least p+1.
        # The cofactor would be at most n/(p+1) ≈ p - 1 + small.
        # So if we trial divide by all primes up to p, we'd find all factors.
        # But we can't enumerate primes up to 10^9.
        
        # Practical limit: trial divide by primes up to 100000 (10^5)
        # After that, the remaining cofactor is either:
        # - 1 (p-smooth)
        # - A prime (check if <= p)
        # - A product of two primes both > 100000
        #   In that case, each is > 100000, so their product > 10^10
        #   But n < p^2 + 201, and for p > 100000, p^2 > 10^10
        #   So this case can happen.
        #   We'd need Pollard-Brent for those.
        
        # Let's use trial division up to 100000, then Miller-Rabin, then Pollard-Brent if needed.
        
        # Generate primes up to 100000 for trial division
        # (already have small_primes up to 31623, let's extend)
        
        # Actually, let me reconsider. The current approach works but is slow.
        # The main bottleneck is calling is_p_smooth for each number.
        
        # Let me optimize is_p_smooth with a higher trial division limit.
        
        k = 0
        for j in range(1, k_max + 1):
            if time.time() - start_time > DEADLINE:
                break
            n = base + j
            if not _is_p_smooth_optimized(n, p, small_primes):
                break
            k = j
        return k
    
    def _is_p_smooth_optimized(n, p, primes_list):
        """Check if all prime factors of n are <= p, using precomputed primes"""
        if n <= 1:
            return True
        
        # Trial divide by primes in primes_list up to min(p, sqrt(n))
        limit = min(p, math.isqrt(n))
        for prime in primes_list:
            if prime > limit:
                break
            if prime > p:
                break
            while n % prime == 0:
                n //= prime
            if n == 1:
                return True
            if n <= p:
                # n is now <= p, so all its factors are <= p
                return True
        
        # n is now either 1, a prime, or a composite with all prime factors > limit
        if n == 1:
            return True
        if n <= p:
            return True
        
        # n > p and n > limit
        # Check if n is prime
        if is_prime_mr(n):
            # n is prime and n > p, so not p-smooth
            return False
        
        # n is composite with factors > limit
        # Use Pollard-Brent
        factor = pollard_brent(n)
        return _is_p_smooth_optimized(factor, p, primes_list) and _is_p_smooth_optimized(n // factor, p, primes_list)
    
    def pollard_brent(n):
        """Pollard's rho with Brent's improvement"""
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(2, n - 1)
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
                        q = q * abs(x - y) % n
                    g = _gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                while True:
                    ys = (ys * ys + c) % n
                    g = _gcd(abs(x - ys), n)
                    if g > 1:
                        break
            if g != n:
                return g
    
    def _gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    # Generate primes up to 100000 for trial division
    trial_primes = sieve(100000)
    
    best_k = 0
    best_p = 2
    
    # Search primes in increasing order
    # Start with small primes, then go larger
    # For efficiency, we can skip composites using Miller-Rabin
    
    # First check small primes
    for p in small_primes:
        if time.time() - start_time > DEADLINE:
            break
        base = p * p
        k = _check_p_smooth_run(base, p, trial_primes, 200)
        if k > best_k:
            best_k = k
            best_p = p
            if k >= 200:
                break
    
    # If we didn't find k=200, search larger primes
    if best_k < 200:
        # Search odd numbers beyond small_primes
        p = small_primes[-1] + 2 if small_primes else 3
        while p <= 1000000000:
            if time.time() - start_time > DEADLINE:
                break
            if is_prime_mr(p):
                base = p * p
                k = _check_p_smooth_run(base, p, trial_primes, 200)
                if k > best_k:
                    best_k = k
                    best_p = p
                    if k >= 200:
                        break
            p += 2
    
    return {'p': best_p, 'k': best_k}

def _check_p_smooth_run(base, p, trial_primes, k_max):
    """Check how many consecutive numbers base+1, base+2, ... are p-smooth"""
    k = 0
    for j in range(1, k_max + 1):
        n = base + j
        if not _is_p_smooth_optimized(n, p, trial_primes):
            break
        k = j
    return k
