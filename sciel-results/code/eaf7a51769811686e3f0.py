"""New program: Randomized prime search with optimized smoothness check."""
# EVOLVE-BLOCK-START

def construct():
    import math
    import random
    import time

    # ---------- Deterministic Miller-Rabin for 64-bit ----------
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
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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

    # ---------- Pollard's Rho (Brent) ----------
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(1, n - 1)
            c = random.randint(1, n - 1)
            m = 128
            g = r = q = 1
            x = ys = y
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
                    g = gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = gcd(abs(x - ys), n)
            if g != n:
                return g

    def is_p_smooth(n, p, small_primes):
        """
        Returns True if n is p-smooth (all prime factors <= p).
        small_primes: list of primes up to some limit for trial division.
        """
        if n == 1:
            return True
        if n <= p:
            return True

        # Trial division by small primes
        temp = n
        for sp in small_primes:
            if sp > p:
                break
            while temp % sp == 0:
                temp //= sp
        if temp == 1:
            return True

        # If temp is prime
        if is_prime(temp):
            return temp <= p

        # If temp is composite, it has a factor.
        # We need to check if all factors are <= p.
        # We can use Pollard to find a factor d.
        # If d > p, return False.
        # If d <= p, divide temp by d and recurse/check again.
        
        # To avoid deep recursion and overhead, we can just factor temp completely
        # and check max factor. But factoring is expensive.
        # Better: check if temp has any factor > p.
        # This is equivalent to: does temp have a prime factor > p?
        
        # Let's try to find a factor.
        # If we find a factor d, and d > p, return False.
        # If d <= p, we can divide it out and continue.
        
        # However, Pollard might return a composite factor.
        # Let's just factor temp fully using a helper.
        
        def has_large_factor(x, limit, small_primes):
            if x == 1:
                return False
            if x <= limit:
                return False
            
            # Trial divide small primes again (optional optimization)
            # We already did this for 'n', but 'x' is a factor of n, 
            # so small primes dividing x also divided n and were removed?
            # No, we removed them from 'n' to get 'temp'. 
            # So 'temp' is not divisible by any small prime in the list.
            # Thus we can skip trial division for 'x'.
            
            if is_prime(x):
                return x > limit

            d = pollard_brent(x)
            if d == x:
                # Should not happen if is_prime is correct and x > 1
                return x > limit
            
            if has_large_factor(d, limit, small_primes):
                return True
            if has_large_factor(x // d, limit, small_primes):
                return True
            return False

        return not has_large_factor(temp, p, small_primes)

    # ---------- Generate Small Primes for Trial Division ----------
    def generate_small_primes(limit):
        sieve = [True] * (limit + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.isqrt(limit)) + 1):
            if sieve[i]:
                for j in range(i*i, limit + 1, i):
                    sieve[j] = False
        return [i for i, is_p in enumerate(sieve) if is_p]

    # ---------- Search ----------
    best_k = 0
    best_p = 2
    
    # Pre-generate small primes for trial division (up to 10,000)
    small_primes = generate_small_primes(10_000)
    
    start_time = time.time()
    TIME_LIMIT = 110.0

    # Define search ranges (low, high)
    # We sample randomly from these ranges
    ranges = [
        (100, 1000),
        (1000, 10_000),
        (10_000, 100_000),
        (100_000, 1_000_000),
        (1_000_000, 5_000_000),
        (5_000_000, 10_000_000)
    ]
    
    # We will perform a number of random trials
    # Each trial picks a random prime from a random range
    # To get primes, we can just pick a random number and check primality,
    # or use a small sieve for the lower ranges and Miller-Rabin for higher.
    
    # For efficiency, let's pre-sieve small ranges and use MR for large.
    # Or just use MR for all? MR is fast for 32-bit.
    
    num_trials = 5000
    
    for _ in range(num_trials):
        if time.time() - start_time > TIME_LIMIT:
            break
        
        # Pick a random range
        low, high = random.choice(ranges)
        # Pick a random number in range
        candidate = random.randint(low, high)
        
        # Ensure it's prime
        if not is_prime(candidate):
            continue
            
        p = candidate
        p2 = p * p
        
        k = 0
        for offset in range(1, 201):
            val = p2 + offset
            if not is_p_smooth(val, p, small_primes):
                break
            k = offset
            
        if k > best_k:
            best_k = k
            best_p = p
            
    # If we found a good k, we can try to refine around best_p?
    # Or just return.
    
    # Fallback if best_k is 0 (should not happen, p=7 gives k=1)
    if best_k == 0:
        # Try small primes explicitly
        for p in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83]:
            p2 = p * p
            k = 0
            for offset in range(1, 201):
                val = p2 + offset
                if not is_p_smooth(val, p, small_primes):
                    break
                k = offset
            if k > best_k:
                best_k = k
                best_p = p

    return {'p': best_p, 'k': best_k}

# EVOLVE-BLOCK-END
