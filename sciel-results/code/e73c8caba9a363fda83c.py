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

    def sieve_primes(limit):
        is_p = [True] * (limit + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                for j in range(i*i, limit+1, i):
                    is_p[j] = False
        return [i for i in range(2, limit+1) if is_p[i]]

    # Precompute small primes for batch division
    SMALL_PRIME_LIMIT = 100000
    small_primes = sieve_primes(SMALL_PRIME_LIMIT)

    def check_run_batch(p, k_max=200):
        """
        Check how many consecutive integers p^2+1, ..., p^2+k_max are p-smooth.
        Uses batch approach: divide out small primes from all numbers at once.
        """
        if time.time() - start_time > DEADLINE:
            return 0
        
        base = p * p
        # For each number base+1 ... base+k_max, we want to check if all prime factors <= p
        # Strategy: 
        # 1. For each small prime q <= min(p, SMALL_PRIME_LIMIT), divide out q from all numbers
        # 2. For remaining cofactors, use Pollard-Brent to factorize and check
        
        # Create array of numbers to check
        nums = [base + i for i in range(1, k_max + 1)]
        # Track which numbers have been fully factored (cofactor = 1)
        # and their remaining cofactor after dividing small primes
        
        # For each small prime q <= p, divide it out from all nums
        # But we can't iterate over all primes up to p if p is large
        # Instead: iterate over small_primes that are <= p
        for q in small_primes:
            if q > p:
                break
            if time.time() - start_time > DEADLINE:
                return 0
            # Find which nums are divisible by q
            # nums[i] = base + i + 1, so we need (base + i + 1) % q == 0
            # i.e., i+1 ≡ -base (mod q), so i+1 ≡ (q - base%q) % q (mod q)
            r = (-base) % q
            # i+1 = r, r+q, r+2q, ... within [1, k_max]
            # i = r-1, r-1+q, r-1+2q, ...
            if r == 0:
                first_i = 0  # i+1 = q, 2q, ... => i = q-1, 2q-1, ...
                # Actually i+1 must be multiple of q: i+1 = q, 2q, ...
                # So i = q-1, 2q-1, ...
                first_i = q - 1
            else:
                first_i = r - 1
            
            # Iterate through multiples
            idx = first_i
            while idx < k_max:
                n = nums[idx]
                # Divide out all factors of q from n
                while n % q == 0:
                    n //= q
                nums[idx] = n
                idx += q
        
        # Now for each remaining number, check if it's 1 (fully factored) or if its prime factors are all <= p
        k = 0
        for i in range(k_max):
            if time.time() - start_time > DEADLINE:
                break
            n = nums[i]
            if n == 1:
                k = i + 1
                continue
            # Check if n has all prime factors <= p
            # First check if n itself is prime
            if is_prime_mr(n):
                if n > p:
                    break
                else:
                    k = i + 1
                    continue
            # Try to factor n using Pollard-Brent
            # But first, try a bit more trial division with primes up to sqrt(n) if feasible
            # Since n could be up to ~p^2, sqrt(n) ~ p, which is too large
            # Use Pollard-Brent
            try:
                factor = pollard_brent(n)
            except Exception:
                break
            
            # Recursively check factor and n//factor
            def all_factors_le_p(m, p_val):
                """Check if all prime factors of m are <= p_val."""
                if m == 1:
                    return True
                if is_prime_mr(m):
                    return m <= p_val
                try:
                    f = pollard_brent(m)
                except Exception:
                    return False
                return all_factors_le_p(f, p_val) and all_factors_le_p(m // f, p_val)
            
            if all_factors_le_p(n, p):
                k = i + 1
            else:
                break
        
        return k

    best_k = 0
    best_p = 2

    # Strategy: Test large primes first, as they're more likely to have long runs
    # Focus on primes in ranges where p^2 is large enough that p-smoothness is likely
    
    # First, try some known good ranges and sample primes
    # Use a segmented sieve to find primes in specific ranges
    
    def find_primes_in_range(lo, hi, max_count=1000):
        """Find up to max_count primes in [lo, hi] using Miller-Rabin."""
        primes = []
        if lo <= 2:
            primes.append(2)
            lo = 3
        if lo % 2 == 0:
            lo += 1
        for candidate in range(lo, hi + 1, 2):
            if len(primes) >= max_count:
                break
            if time.time() - start_time > DEADLINE:
                break
            if is_prime_mr(candidate):
                primes.append(candidate)
        return primes

    # Test ranges from large to small, focusing on where long runs are likely
    # Based on the structure, larger p gives more "room" for smooth numbers
    search_ranges = [
        (800000000, 1000000000),
        (500000000, 800000000),
        (200000000, 500000000),
        (50000000, 200000000),
        (10000000, 50000000),
        (1000000, 10000000),
        (100000, 1000000),
        (10000, 100000),
        (1000, 10000),
        (100, 1000),
    ]

    for lo, hi in search_ranges:
        if time.time() - start_time > DEADLINE:
            break
        if best_k >= 200:
            break
        
        # Sample primes in this range
        # Use a step to avoid testing every odd number
        range_size = hi - lo
        # Test at most ~500 primes per range to stay within time
        step = max(2, range_size // 500)
        if step % 2 == 0:
            step += 1  # keep odd
        
        for p in range(lo, hi + 1, step):
            if time.time() - start_time > DEADLINE:
                break
            if p < 2:
                continue
            if p % 2 == 0:
                continue
            if not is_prime_mr(p):
                continue
            
            k = check_run_batch(p, 200)
            if k > best_k:
                best_k = k
                best_p = p
                if best_k >= 200:
                    break
        
        if best_k >= 200:
            break

    # If we still have time, do a finer search around the best found
    if best_k < 200 and time.time() - start_time < DEADLINE - 10:
        # Search near best_p for potentially better primes
        if best_p > 10:
            search_lo = max(2, best_p - 100000)
            search_hi = best_p + 100000
            step = 2
            for p in range(search_lo, search_hi + 1, step):
                if time.time() - start_time > DEADLINE:
                    break
                if p < 2 or p % 2 == 0:
                    continue
                if not is_prime_mr(p):
                    continue
                if p == best_p:
                    continue
                k = check_run_batch(p, 200)
                if k > best_k:
                    best_k = k
                    best_p = p
                    if best_k >= 200:
                        break

    return {'p': best_p, 'k': best_k}
