import time
import random
import math

def is_prime(n):
    if n < 2:
        return False
    # Small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n % p == 0:
            return n == p
    # Miller-Rabin
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # Deterministic bases for n < 2^64
    # For n < 3,474,749,660,383, bases 2,3,5,7,11,13,17 are enough.
    # For n < 10^9, even fewer are enough.
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

def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    y, c, m = random.randrange(1, n), random.randrange(1, n), 128
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
    return g

def largest_prime_factor(n, limit):
    # Returns the largest prime factor of n, or limit+1 if there's a factor > limit
    # We want to check if all factors <= limit.
    # So if we find a factor > limit, return limit+1.
    # If we fully factorize and all are <= limit, return the max factor.
    
    # Trial division by small primes
    # We can precompute primes up to 10^6
    # But for simplicity, let's just trial divide by 2, 3, 5, ... up to 100
    # Then use Pollard-Brent for the rest.
    
    factors = []
    d = 2
    # Quick check for small factors
    while d * d <= n and d < 100:
        while n % d == 0:
            if d > limit:
                return limit + 1
            factors.append(d)
            n //= d
        d += 1 if d == 2 else 2 # 2, 3, 5, 7...
        
    if n > 1:
        # Factor the remaining n
        # If n is prime, it's a factor.
        # If n is composite, find a factor.
        # We need to ensure all factors are <= limit.
        
        # If n <= limit, it's fine (either prime or product of smalls, but we already removed smalls).
        # Actually, if n <= limit, we don't need to factor it further to check the bound, 
        # unless we want the exact max factor. But we just need to know if ANY factor > limit.
        # If n <= limit, then all its factors are <= n <= limit. So it's safe.
        if n <= limit:
             return max(factors + [n]) if factors else n
             
        # If n > limit, we must factor it to see if it has a factor > limit.
        # If n is prime, then n > limit, so return limit+1.
        # If n is composite, it might have factors <= limit and one > limit, or all <= limit.
        
        # Use Pollard-Brent to find a factor
        # We need to recursively factor n.
        def factorize(m, lim):
            if m == 1:
                return []
            if m <= lim:
                return [m]
            if is_prime(m):
                return [m]
            
            d = pollard_brent(m)
            return factorize(d, lim) + factorize(m // d, lim)
            
        rem_factors = factorize(n, limit)
        for f in rem_factors:
            if f > limit:
                return limit + 1
        factors.extend(rem_factors)
        
    return max(factors) if factors else 1

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_k = 0
    best_p = 2
    
    # Precompute small primes for trial division in is_prime? 
    # is_prime handles smalls.
    
    # Search range
    # Let's try random primes in [10^6, 10^9]
    # Or sequential?
    # Random is better for finding outliers.
    
    # To generate random primes, we can pick random odd numbers and test.
    # Density of primes near 10^9 is 1/log(10^9) ~ 1/20.
    # So we test ~20 numbers to find a prime.
    
    # Let's do a mix: some sequential from 10^6, some random.
    
    # Actually, just random sampling in a wide range is good.
    # Let's sample from [10^5, 10^9].
    
    # Optimization: 
    # If p is small, p^2 is small.
    # If p is large, p^2 is large.
    
    # Let's iterate.
    
    # We can speed up smoothness check by checking gcd with product of primes? No.
    
    # Let's just run the loop.
    
    # To avoid bias, let's use a random seed.
    random.seed(42)
    
    # We'll check 1000 random primes.
    # Each check involves factoring ~k numbers.
    # If k is small, it's fast.
    
    # Let's also try some specific small primes that might have good runs?
    # The seed program checked up to 2000.
    # Let's include a quick scan of small primes to be safe, then large random.
    
    # Small primes scan (fast)
    for p in range(3, 1000, 2):
        if time.time() > deadline:
            break
        if not is_prime(p):
            continue
        k = 0
        while True:
            n = p * p + k + 1
            max_f = largest_prime_factor(n, p)
            if max_f <= p:
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
            
    # Large primes random search
    # We will try to find primes in [10^6, 10^9]
    # Number of trials: as many as time permits.
    
    # To speed up, we can check if p^2+1 is p-smooth first.
    # If not, k=0.
    
    # Let's refine largest_prime_factor to be faster.
    # The current one calls is_prime which is a bit heavy.
    # But for n ~ 10^18, is_prime is fast.
    
    # Let's run random search.
    while time.time() < deadline:
        # Pick a random candidate
        # Range 10^6 to 10^9
        p = random.randint(10**6, 10**9)
        if p % 2 == 0:
            p += 1
        # Check primality
        if not is_prime(p):
            continue
            
        # Check smoothness run
        k = 0
        while True:
            n = p * p + k + 1
            max_f = largest_prime_factor(n, p)
            if max_f <= p:
                k += 1
                # Cap k to avoid infinite loop if something goes wrong, 
                # but theoretically k is small.
                if k > 200:
                    break
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p
            
    return {'p': best_p, 'k': best_k}
