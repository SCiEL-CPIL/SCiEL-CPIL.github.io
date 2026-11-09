"""Segmented-sieve approach for long p-smooth runs after p^2."""
import time
import math
import random

def _is_prime_mr(n):
    """Deterministic Miller-Rabin for n < 3.3e24."""
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

def _sieve_primes(limit):
    """Sieve of Eratosthenes, returns list of primes up to limit."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def _segmented_spf_interval(lo, hi, primes):
    """
    For the interval [lo, hi], compute for each number the largest prime factor.
    Returns a list of length (hi - lo + 1) where entry i holds the largest
    prime factor of (lo + i), or 0 if the number is 1.
    
    Uses segmented sieve: for each prime p, mark multiples in [lo, hi].
    The largest prime factor is tracked by overwriting with larger primes.
    """
    length = hi - lo + 1
    # largest_prime_factor[i] = largest prime factor of (lo+i), 0 initially
    lpf = [0] * length
    
    # Sieve primes up to sqrt(hi)
    sqrt_hi = int(math.isqrt(hi)) + 1
    # Generate primes up to sqrt_hi using simple sieve
    if sqrt_hi >= 2:
        spf_sieve = bytearray(b'\x01') * (sqrt_hi + 1)
        spf_sieve[0] = spf_sieve[1] = 0
        for i in range(2, int(math.isqrt(sqrt_hi)) + 1):
            if spf_sieve[i]:
                spf_sieve[i*i::i] = bytearray(len(spf_sieve[i*i::i]))
        small_primes = [i for i in range(2, sqrt_hi + 1) if spf_sieve[i]]
    else:
        small_primes = []
    
    for p in small_primes:
        # Find the first multiple of p in [lo, hi]
        start = ((lo + p - 1) // p) * p
        for m in range(start, hi + 1, p):
            idx = m - lo
            if p > lpf[idx]:
                lpf[idx] = p
    
    # For numbers in [lo, hi] that still have lpf == 0, they are either 1 or prime
    # If lpf[idx] == 0 and (lo+idx) > 1, then (lo+idx) is prime (since all
    # prime factors up to sqrt(hi) were checked). But we need to verify.
    # Actually, if (lo+idx) has no factor <= sqrt(hi), it's prime (or 1).
    # But it could be a product of two primes both > sqrt(hi)? No, because
    # if n = a*b with a <= b, then a <= sqrt(n) <= sqrt(hi).
    # So if no prime <= sqrt(hi) divides n, then n is prime (or n==1).
    for idx in range(length):
        val = lo + idx
        if val <= 1:
            lpf[idx] = 0
        elif lpf[idx] == 0:
            # val is prime (or 1, handled above)
            lpf[idx] = val
    
    return lpf

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    # Precompute small primes for the segmented sieve
    # We need primes up to sqrt(p^2 + 200) ≈ p for p up to 10^9
    # But we can't sieve up to 10^9. Instead, we sieve up to a manageable limit
    # and handle the rest.
    
    # Strategy: For a given p, the interval is [p^2+1, p^2+200].
    # sqrt(p^2+200) ≈ p. We need primes up to p to do segmented sieve.
    # For small p (say p <= 100000), we can precompute all primes up to p.
    # For larger p, we need a different approach.
    
    # Better approach: Precompute primes up to some limit L.
    # For p <= L, we can use the full list of primes up to p for segmented sieve.
    # For p > L, we need primes up to p, which is expensive.
    
    # Alternative: Use a hybrid approach.
    # For the segmented sieve of [p^2+1, p^2+200], we only need primes up to
    # sqrt(p^2+200) ≈ p. But we can be smarter:
    # A number n in [p^2+1, p^2+200] is p-smooth iff all its prime factors <= p.
    # Since n ≈ p^2, any prime factor q of n satisfies q <= n ≈ p^2.
    # If q > p, then n/q < p, so n/q is a small number.
    # Actually, let's just do the segmented sieve properly.
    
    # For the segmented sieve, we need all primes up to sqrt(hi) where hi = p^2+200.
    # sqrt(p^2+200) ≈ p + 200/(2p) ≈ p for large p.
    # So we need primes up to ~p.
    
    # Precompute primes up to a reasonable limit for the sieve.
    # Let's precompute primes up to 1_000_000 first.
    SIEVE_LIMIT = 1_000_000
    primes = _sieve_primes(SIEVE_LIMIT)
    
    best_k = 0
    best_p = 2
    
    # Phase 1: Search primes up to SIEVE_LIMIT using segmented sieve
    # For each prime p, compute lpf for [p^2+1, p^2+200] and check max(lpf) <= p
    
    # We can search all primes up to SIEVE_LIMIT
    for p in primes:
        if time.time() - start_time > DEADLINE:
            break
        
        p2 = p * p
        lo = p2 + 1
        hi = p2 + 200
        
        # For the segmented sieve, we need primes up to sqrt(hi)
        sqrt_hi = int(math.isqrt(hi)) + 1
        
        # If sqrt_hi <= SIEVE_LIMIT, we have all needed primes
        if sqrt_hi <= SIEVE_LIMIT:
            # Use primes up to sqrt_hi
            needed_primes = [q for q in primes if q <= sqrt_hi]
            lpf = _segmented_spf_interval(lo, hi, needed_primes)
            
            # Check: all numbers p^2+1 ... p^2+k must have all prime factors <= p
            # i.e., max prime factor of each number <= p
            k = 0
            for i in range(200):
                if lpf[i] > p:
                    break
                k += 1
            
            if k > best_k:
                best_k = k
                best_p = p
        else:
            # sqrt_hi > SIEVE_LIMIT, need more primes
            # This happens when p > ~1_000_000
            # For now, skip or use a fallback
            # We'll handle larger p in Phase 2
            pass
    
    # Phase 2: Search larger primes (SIEVE_LIMIT < p <= 10^9)
    # For these, we need a different strategy since we can't easily get primes up to p.
    # Use random sampling with a faster check.
    
    # For large p, p^2 + i is ~p^2. The prime factors of p^2+i are either
    # small (<= some bound) or large. We can do trial division by small primes
    # and then check the remainder.
    
    # Precompute primes up to a smaller limit for trial division
    SMALL_LIMIT = 100_000
    small_primes = _sieve_primes(SMALL_LIMIT)
    
    random.seed(42)
    
    # Try random primes in various ranges
    search_ranges = [
        (1_000_000, 5_000_000),
        (5_000_000, 20_000_000),
        (20_000_000, 100_000_000),
        (100_000_000, 500_000_000),
        (500_000_000, 1_000_000_000),
    ]
    
    for lo_r, hi_r in search_ranges:
        if time.time() - start_time > DEADLINE:
            break
        
        max_tries = 200
        tried = 0
        while tried < max_tries and time.time() - start_time < DEADLINE:
            tried += 1
            p = random.randint(lo_r, hi_r)
            if p % 2 == 0:
                p += 1
            if p < 2 or p > 10**9:
                continue
            if not _is_prime_mr(p):
                continue
            
            p2 = p * p
            k = 0
            for i in range(1, 201):
                if time.time() - start_time > DEADLINE:
                    break
                n = p2 + i
                # Check if n is p-smooth
                # Trial divide by small primes
                tmp = n
                valid = True
                for sp in small_primes:
                    if sp > p:
                        break
                    if sp * sp > tmp:
                        break
                    while tmp % sp == 0:
                        tmp //= sp
                if tmp > 1:
                    # tmp has a factor > SMALL_LIMIT
                    # Check if tmp is prime
                    if _is_prime_mr(tmp):
                        if tmp > p:
                            valid = False
                    else:
                        # tmp is composite with all factors > SMALL_LIMIT
                        # At least one factor > p (since tmp > 1 and factors > SMALL_LIMIT)
                        # Actually, tmp could be a product of primes all <= p but > SMALL_LIMIT
                        # We need to factor tmp to check
                        # Use Pollard-Brent
                        def _pollard_brent(n):
                            if n % 2 == 0:
                                return 2
                            if n % 3 == 0:
                                return 3
                            while True:
                                y = random.randint(2, n - 1)
                                c = random.randint(1, n - 1)
                                m = 128
                                g = r = q = 1
                                x = ys = y
                                while g == 1:
                                    x = y
                                    for _ in range(r):
                                        y = (y * y + c) % n
                                    k_inner = 0
                                    while k_inner < r and g == 1:
                                        ys = y
                                        for _ in range(min(m, r - k_inner)):
                                            y = (y * y + c) % n
                                            q = q * abs(x - y) % n
                                        g = math.gcd(q, n)
                                        k_inner += m
                                    r *= 2
                                if g == n:
                                    while True:
                                        ys = (ys * ys + c) % n
                                        g = math.gcd(abs(x - ys), n)
                                        if g > 1:
                                            break
                                if g != n:
                                    return g
                        
                        # Factor tmp
                        stack = [tmp]
                        all_leq_p = True
                        while stack:
                            m = stack.pop()
                            if m == 1:
                                continue
                            if _is_prime_mr(m):
                                if m > p:
                                    all_leq_p = False
                                    break
                            else:
                                try:
                                    d = _pollard_brent(m)
                                    stack.append(d)
                                    stack.append(m // d)
                                except:
                                    all_leq_p = False
                                    break
                        if not all_leq_p:
                            valid = False
                
                if not valid:
                    break
                k += 1
            
            if k > best_k:
                best_k = k
                best_p = p
    
    # Phase 3: If time remains, do a more thorough search near promising ranges
    # Also try sequential search around best_p
    if time.time() - start_time < DEADLINE - 20:
        # Search primes near best_p
        search_lo = max(2, best_p - 50000)
        search_hi = min(10**9, best_p + 50000)
        
        # Generate primes in this range using segmented sieve
        length = search_hi - search_lo + 1
        if length > 0:
            # Sieve for primes in [search_lo, search_hi]
            sqrt_hi = int(math.isqrt(search_hi)) + 1
            # Get primes up to sqrt_hi
            if sqrt_hi <= SIEVE_LIMIT:
                sieve_primes_for_range = [q for q in primes if q <= sqrt_hi]
            else:
                # Need to generate primes up to sqrt_hi
                if sqrt_hi >= 2:
                    spf_sieve = bytearray(b'\x01') * (sqrt_hi + 1)
                    spf_sieve[0] = spf_sieve[1] = 0
                    for i in range(2, int(math.isqrt(sqrt_hi)) + 1):
                        if spf_sieve[i]:
                            spf_sieve[i*i::i] = bytearray(len(spf_sieve[i*i::i]))
                    sieve_primes_for_range = [i for i in range(2, sqrt_hi + 1) if spf_sieve[i]]
                else:
                    sieve_primes_for_range = []
            
            # Segmented sieve for [search_lo, search_hi]
            is_prime_arr = bytearray(b'\x01') * length
            for q in sieve_primes_for_range:
                start = ((search_lo + q - 1) // q) * q
                if start == q:
                    start += q
                for m in range(start, search_hi + 1, q):
                    is_prime_arr[m - search_lo] = 0
            
            # Check each prime in range
            for offset in range(length):
                if time.time() - start_time > DEADLINE:
                    break
                p = search_lo + offset
                if p < 2:
                    continue
                if not is_prime_arr[offset]:
                    continue
                if p > 10**9:
                    continue
                
                p2 = p * p
                lo = p2 + 1
                hi = p2 + 200
                sqrt_hi_val = int(math.isqrt(hi)) + 1
                
                if sqrt_hi_val <= SIEVE_LIMIT:
                    needed_primes = [q for q in primes if q <= sqrt_hi_val]
                    lpf = _segmented_spf_interval(lo, hi, needed_primes)
                    k = 0
                    for i in range(200):
                        if lpf[i] > p:
                            break
                        k += 1
                    if k > best_k:
                        best_k = k
                        best_p = p
                else:
                    # Fallback: trial division
                    k = 0
                    for i in range(1, 201):
                        if time.time() - start_time > DEADLINE:
                            break
                        n = p2 + i
                        tmp = n
                        valid = True
                        for sp in small_primes:
                            if sp > p:
                                break
                            if sp * sp > tmp:
                                break
                            while tmp % sp == 0:
                                tmp //= sp
                        if tmp > 1:
                            if _is_prime_mr(tmp):
                                if tmp > p:
                                    valid = False
                            else:
                                # Need to factor
                                def _pollard_brent2(n):
                                    if n % 2 == 0:
                                        return 2
                                    if n % 3 == 0:
                                        return 3
                                    while True:
                                        y = random.randint(2, n - 1)
                                        c = random.randint(1, n - 1)
                                        m = 128
                                        g = r = q = 1
                                        x = ys = y
                                        while g == 1:
                                            x = y
                                            for _ in range(r):
                                                y = (y * y + c) % n
                                            k_inner = 0
                                            while k_inner < r and g == 1:
                                                ys = y
                                                for _ in range(min(m, r - k_inner)):
                                                    y = (y * y + c) % n
                                                    q = q * abs(x - y) % n
                                                g = math.gcd(q, n)
                                                k_inner += m
                                            r *= 2
                                        if g == n:
                                            while True:
                                                ys = (ys * ys + c) % n
                                                g = math.gcd(abs(x - ys), n)
                                                if g > 1:
                                                    break
                                        if g != n:
                                            return g
                                
                                stack = [tmp]
                                all_leq_p = True
                                while stack:
                                    m = stack.pop()
                                    if m == 1:
                                        continue
                                    if _is_prime_mr(m):
                                        if m > p:
                                            all_leq_p = False
                                            break
                                    else:
                                        try:
                                            d = _pollard_brent2(m)
                                            stack.append(d)
                                            stack.append(m // d)
                                        except:
                                            all_leq_p = False
                                            break
                                if not all_leq_p:
                                    valid = False
                        if not valid:
                            break
                        k += 1
                    if k > best_k:
                        best_k = k
                        best_p = p
    
    if best_k < 1:
        best_k = 1
        best_p = 2
    
    return {'p': best_p, 'k': best_k}
