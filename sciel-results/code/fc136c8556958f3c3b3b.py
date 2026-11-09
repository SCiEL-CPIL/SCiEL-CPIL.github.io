"""Optimized program: Focus on promising prime ranges with fast smoothness check."""
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
        if n % 5 == 0:
            return 5
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

    def is_p_smooth(n, p):
        """
        Returns True if all prime factors of n are <= p.
        Uses trial division by primes up to min(p, sqrt(n)), then checks remainder.
        """
        if n <= 1:
            return True
        if n <= p:
            return True
        
        # Trial division by small primes up to min(p, isqrt(n))
        limit = min(p, int(math.isqrt(n)))
        temp = n
        # Check 2 separately
        while temp % 2 == 0:
            temp //= 2
        if temp == 1:
            return True
        # Check odd primes up to limit
        d = 3
        while d <= limit and d * d <= temp:
            while temp % d == 0:
                temp //= d
            if temp == 1:
                return True
            d += 2
        if temp == 1:
            return True
        # If temp is 1, done. If temp <= p, done.
        if temp <= p:
            return True
        # If temp is prime, check if <= p
        if is_prime(temp):
            return temp <= p
        # Otherwise, temp is composite and > p, so it has a factor > p
        # But we need to verify: could temp be a product of primes all <= p?
        # If d*d > temp, then temp is prime. Otherwise, we didn't fully factor.
        # Since we stopped at d where d*d > temp or d > limit, if temp > 1 and temp > p,
        # and we can't factor it with primes <= min(p, sqrt(temp)), then:
        # If temp <= p, we'd have returned True.
        # If temp > p, it could be prime > p (fail) or composite with all factors <= p.
        # But if all factors of temp are <= p, and temp > p, then temp must have at least
        # two factors, each <= p. But we tried all primes up to min(p, sqrt(temp)).
        # If sqrt(temp) <= p, we tried all primes up to sqrt(temp), so if temp is composite,
        # we would have found a factor. So if we get here with temp > 1 and temp > p,
        # and we exhausted primes up to min(p, sqrt(temp)), then:
        # - If sqrt(temp) <= p: temp is prime (since no factor found up to sqrt(temp)), so temp > p means fail.
        # - If sqrt(temp) > p: we only tried up to p. Temp could be composite with factors > p.
        # In either case, if temp > p and we couldn't reduce it to 1 using primes <= min(p, sqrt(temp)),
        # we need to check if temp itself is <= p (already checked) or if it's prime <= p.
        # Actually, the safest is: if temp > p, return False unless we can prove all factors <= p.
        # Given the constraints, if temp > p after trial division by all primes <= min(p, sqrt(n)),
        # then temp has a prime factor > p.
        return False

    # ---------- Sieve of Eratosthenes for small primes ----------
    def sieve_primes(n):
        if n < 2:
            return []
        is_prime_arr = [True] * (n + 1)
        is_prime_arr[0] = is_prime_arr[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime_arr[i]:
                for j in range(i*i, n + 1, i):
                    is_prime_arr[j] = False
        return [i for i in range(2, n + 1) if is_prime_arr[i]]

    # ---------- Search ----------
    best_k = 0
    best_p = 7  # p=7 gives k=1 as fallback
    best_k = 1

    start_time = time.time()
    TIME_LIMIT = 110.0

    # Precompute primes up to 100000 for trial division
    small_primes = sieve_primes(100000)
    
    def is_p_smooth_fast(n, p, primes_list):
        """
        Fast check: trial divide by primes in primes_list up to min(p, sqrt(n)).
        """
        if n <= 1:
            return True
        if n <= p:
            return True
        
        temp = n
        limit = min(p, int(math.isqrt(n)))
        for prime in primes_list:
            if prime > limit:
                break
            while temp % prime == 0:
                temp //= prime
            if temp == 1:
                return True
        if temp == 1:
            return True
        if temp <= p:
            return True
        # Check if temp is prime
        if is_prime(temp):
            return temp <= p
        # temp is composite and > p, so has factor > p
        return False

    # Search strategy:
    # 1. Check small primes (2 to 10000) thoroughly
    # 2. Check medium primes (10000 to 100000) with faster check
    # 3. If time, check larger ranges
    
    # Phase 1: Small primes up to 10000
    primes_small = sieve_primes(10000)
    for p in primes_small:
        if time.time() - start_time > TIME_LIMIT:
            break
        p2 = p * p
        k = 0
        for offset in range(1, 201):
            val = p2 + offset
            if not is_p_smooth_fast(val, p, small_primes):
                break
            k = offset
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Medium primes 10000 to 200000
    if time.time() - start_time < TIME_LIMIT:
        # Generate primes in [10000, 200000]
        primes_medium = [p for p in sieve_primes(200000) if p >= 10000]
        for p in primes_medium:
            if time.time() - start_time > TIME_LIMIT:
                break
            p2 = p * p
            # Quick check: if p^2+1 is not p-smooth, skip (k=0)
            if not is_p_smooth_fast(p2 + 1, p, small_primes):
                continue
            k = 0
            for offset in range(1, 201):
                val = p2 + offset
                if not is_p_smooth_fast(val, p, small_primes):
                    break
                k = offset
            if k > best_k:
                best_k = k
                best_p = p

    # Phase 3: Larger primes if time permits
    if time.time() - start_time < TIME_LIMIT - 20:
        # Use segmented sieve for larger ranges
        def segmented_sieve(low, high):
            size = high - low + 1
            sieve = [True] * size
            limit = int(math.isqrt(high))
            small_primes_local = sieve_primes(limit)
            for p in small_primes_local:
                start = ((low + p - 1) // p) * p
                if start == p:
                    start += p
                for j in range(start, high + 1, p):
                    sieve[j - low] = False
            return [low + i for i in range(size) if sieve[i]]
        
        # Check a few segments
        for start_seg in [200000, 500000, 1000000]:
            if time.time() - start_time > TIME_LIMIT - 10:
                break
            end_seg = min(start_seg + 100000, 2000000)
            primes_seg = segmented_sieve(start_seg, end_seg)
            for p in primes_seg:
                if time.time() - start_time > TIME_LIMIT:
                    break
                p2 = p * p
                if not is_p_smooth_fast(p2 + 1, p, small_primes):
                    continue
                k = 0
                for offset in range(1, 201):
                    val = p2 + offset
                    if not is_p_smooth_fast(val, p, small_primes):
                        break
                    k = offset
                if k > best_k:
                    best_k = k
                    best_p = p

    return {'p': best_p, 'k': best_k}

# EVOLVE-BLOCK-END
