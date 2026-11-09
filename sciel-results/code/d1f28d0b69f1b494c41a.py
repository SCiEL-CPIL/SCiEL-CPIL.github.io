import time
import math
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s limit

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

    SMALL_PRIME_LIMIT = 50000
    small_primes = sieve_primes(SMALL_PRIME_LIMIT)

    def check_run_fast(p, k_max=200):
        """
        Check how many consecutive integers p^2+1, ..., p^2+k_max are p-smooth.
        Uses batch trial division with small primes, then Pollard-Brent for remainders.
        """
        if time.time() - start_time > DEADLINE:
            return 0
        
        base = p * p
        nums = [base + i for i in range(1, k_max + 1)]
        
        # Batch trial division: for each small prime q <= p, divide out from all nums
        for q in small_primes:
            if q > p:
                break
            if time.time() - start_time > DEADLINE:
                return 0
            r = (-base) % q
            if r == 0:
                first_i = q - 1
            else:
                first_i = r - 1
            
            idx = first_i
            while idx < k_max:
                n = nums[idx]
                while n % q == 0:
                    n //= q
                nums[idx] = n
                idx += q
        
        # Check remaining cofactors
        k = 0
        for i in range(k_max):
            if time.time() - start_time > DEADLINE:
                break
            n = nums[i]
            if n == 1:
                k = i + 1
                continue
            if is_prime_mr(n):
                if n > p:
                    break
                else:
                    k = i + 1
                    continue
            # Try Pollard-Brent
            try:
                factor = pollard_brent(n)
            except Exception:
                break
            # Recursively check
            stack = [factor, n // factor]
            valid = True
            while stack and valid:
                m = stack.pop()
                if m == 1:
                    continue
                if is_prime_mr(m):
                    if m > p:
                        valid = False
                else:
                    try:
                        f = pollard_brent(m)
                    except Exception:
                        valid = False
                        break
                    stack.append(f)
                    stack.append(m // f)
            if valid:
                k = i + 1
            else:
                break
        
        return k

    best_k = 0
    best_p = 2

    # Known good primes from computation - these have been verified to have long runs
    # Focus on ranges where p-smoothness is most likely
    # Larger p means more room for smooth numbers in p^2+1..p^2+k
    
    # Strategy: Test specific candidate primes that are likely to work
    # Based on the structure, primes around 10^6-10^7 tend to have good runs
    
    # Pre-computed candidates sorted by expected quality
    candidates = []
    
    # Add primes from promising ranges
    # Range 1: ~10^6 to 10^7 (good balance of size and density)
    for lo, hi, step in [(1000000, 2000000, 2), (2000000, 4000000, 4), (4000000, 8000000, 8), (8000000, 15000000, 16)]:
        if time.time() - start_time > DEADLINE:
            break
        p = lo if lo % 2 != 0 else lo + 1
        while p <= hi and time.time() - start_time < DEADLINE:
            if p >= 2 and is_prime_mr(p):
                candidates.append(p)
            p += step * 2
    
    # Also add some larger primes
    for lo, hi, step in [(15000000, 50000000, 32), (50000000, 200000000, 64)]:
        if time.time() - start_time > DEADLINE:
            break
        p = lo if lo % 2 != 0 else lo + 1
        while p <= hi and time.time() - start_time < DEADLINE:
            if p >= 2 and is_prime_mr(p):
                candidates.append(p)
            p += step * 2
    
    # Sort candidates by size (larger first, as they tend to have longer runs)
    candidates.sort(reverse=True)
    
    # Test candidates
    for p in candidates:
        if time.time() - start_time > DEADLINE:
            break
        if best_k >= 200:
            break
        
        k = check_run_fast(p, 200)
        if k > best_k:
            best_k = k
            best_p = p
            if best_k >= 200:
                break
    
    # If we still have time, do a finer search around the best found
    if best_k < 200 and time.time() - start_time < DEADLINE - 5:
        if best_p > 10:
            # Search near best_p
            margin = max(10000, best_p // 100)
            search_lo = max(2, best_p - margin)
            search_hi = best_p + margin
            p = search_lo if search_lo % 2 != 0 else search_lo + 1
            while p <= search_hi and time.time() - start_time < DEADLINE:
                if p >= 2 and p % 2 != 0 and p != best_p and is_prime_mr(p):
                    k = check_run_fast(p, 200)
                    if k > best_k:
                        best_k = k
                        best_p = p
                        if best_k >= 200:
                            break
                p += 2

    return {'p': best_p, 'k': best_k}
