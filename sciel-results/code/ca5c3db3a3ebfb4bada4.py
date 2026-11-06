import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # 1. Deterministic Miller-Rabin for 64-bit integers
    def is_prime(n):
        if n < 2:
            return False
        if n in (2, 3):
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        # Bases sufficient for n < 3,317,044,064,679,887,385,961,981
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    # 2. Pollard-Brent
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

    # 3. Generate primes up to 10^7 using a simple sieve
    limit = 10**7
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    primes = [i for i, is_prime_flag in enumerate(sieve) if is_prime_flag]
    
    # Precompute small primes for trial division (up to 100,000)
    # This covers most small factors quickly.
    small_primes = [q for q in primes if q <= 100000]

    # 4. Helper: Check if n is p-smooth
    def is_p_smooth(n, p):
        if n <= 1:
            return True
        if n <= p:
            return True
        
        # Divide by small primes
        for q in small_primes:
            if q > p:
                break
            if q * q > n:
                break
            if n % q == 0:
                while n % q == 0:
                    n //= q
                if n == 1:
                    return True
                if n <= p:
                    return True
                if is_prime(n):
                    return n <= p
                # If n is composite and > p, we break to use Pollard-Brent
                break
        
        if n == 1:
            return True
        if n <= p:
            return True
            
        # Check if remainder is prime
        if is_prime(n):
            return n <= p
        
        # Use Pollard-Brent to find a factor
        def check_factor(m, p):
            if m == 1:
                return True
            if m <= p:
                return True
            if is_prime(m):
                return m <= p
            
            f = None
            for _ in range(5):
                f = pollard_brent(m)
                if f is not None and 1 < f < m:
                    break
            if f is None or f == m:
                return False
            
            return check_factor(f, p) and check_factor(m // f, p)
            
        return check_factor(n, p)

    best_k = 0
    best_p = 0
    
    # Main search loop
    # Scan primes up to 10^7
    for p in primes:
        if time.time() > deadline:
            break
            
        if p < 10:
            continue
            
        # Quick check for p^2+1
        n1 = p * p + 1
        if not is_p_smooth(n1, p):
            continue
            
        # If p^2+1 is smooth, check further
        k = 1
        for i in range(2, 201):
            if time.time() > deadline:
                break
            n = p * p + i
            if not is_p_smooth(n, p):
                break
            k += 1
            
        if k > best_k:
            best_k = k
            best_p = p
            
    # If time remains, try to search larger primes using a segmented approach or random sampling
    # For now, we rely on the 10^7 scan. 
    # To improve, we could scan 10^7 to 2*10^7 if time permits.
    
    if time.time() < deadline - 10:
        # Generate primes in [10^7, 2*10^7]
        # Simple segmented sieve
        start = 10**7 + 1
        end = 2 * 10**7
        # We need primes up to sqrt(end) for sieving
        sqrt_end = int(end**0.5) + 1
        # Re-use small primes for sieving
        seg_sieve = bytearray(b'\x01') * (end - start + 1)
        for p in small_primes:
            if p * p > end:
                break
            # Find first multiple of p in [start, end]
            first = ((start + p - 1) // p) * p
            if first == p:
                first += p
            for j in range(first, end + 1, p):
                seg_sieve[j - start] = 0
        
        # Check candidates in this segment
        for i in range(start, end + 1):
            if time.time() > deadline:
                break
            if seg_sieve[i - start]:
                if is_prime(i):
                    p = i
                    n1 = p * p + 1
                    if not is_p_smooth(n1, p):
                        continue
                    k = 1
                    for j in range(2, 201):
                        if time.time() > deadline:
                            break
                        n = p * p + j
                        if not is_p_smooth(n, p):
                            break
                        k += 1
                    if k > best_k:
                        best_k = k
                        best_p = p

    return {'p': best_p, 'k': best_k}
