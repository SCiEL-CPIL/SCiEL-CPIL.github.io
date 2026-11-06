import math
import time
import random

def construct():
    deadline = time.time() + 140.0  # Safety margin within 150s limit
    
    # 1. Sieve primes up to 10^6 for trial division
    SIEVE_LIMIT = 10**6
    is_prime_arr = bytearray(b'\x01') * (SIEVE_LIMIT + 1)
    is_prime_arr[0:2] = b'\x00\x00'
    for i in range(2, int(SIEVE_LIMIT**0.5) + 1):
        if is_prime_arr[i]:
            is_prime_arr[i*i::i] = bytearray(len(is_prime_arr[i*i::i]))
    primes = [i for i in range(2, SIEVE_LIMIT + 1) if is_prime_arr[i]]
    
    # Precompute small primes for quick trial division in smoothness check
    SMALL_PRIMES = primes[:200]  # Primes up to ~1223

    def is_prime_mr(n):
        if n < 2: return False
        if n < 4: return True
        if n % 2 == 0 or n % 3 == 0: return False
        for sp in [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n == sp: return True
            if n % sp == 0: return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if a >= n: continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1: continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1: break
            else:
                return False
        return True

    def pollard_brent(n):
        if n % 2 == 0: return 2
        if n % 3 == 0: return 3
        if n % 5 == 0: return 5
        y, c, m = random.randint(1, n-1), random.randint(1, n-1), 128
        g, r, q = 1, 1, 1
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r-k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g > 1: break
        return g

    def is_p_smooth(n, p):
        """
        Check if n is p-smooth.
        """
        if n <= 1: return True
        
        # Trial divide by small primes
        for q in SMALL_PRIMES:
            if q > p:
                break
            if q * q > n:
                break
            while n % q == 0:
                n //= q
        
        if n == 1:
            return True
            
        # If p <= 10^6, we have divided by all primes <= p (if n > 1, it has factor > p)
        # But wait, SMALL_PRIMES only goes up to ~1223.
        # If p > 1223 and p <= 10^6, we need to divide by primes up to p.
        # However, if n is large, and p is small, we might stop early.
        
        # Optimization: if p is small, we must ensure all factors <= p.
        # If p <= 10^6, we can use the full primes list up to p.
        # If p > 10^6, we use primes up to 10^6, then handle remainder.
        
        # Let's refine the trial division to use the full primes list if p is small enough
        if p <= SIEVE_LIMIT:
            # Use primes list up to p
            # Find index of p in primes
            # Since primes is sorted, we can binary search or just iterate
            # Iterating 200 small primes was a shortcut. Let's do it properly for p <= 10^6
            # But iterating 78k primes for every n is slow.
            # Actually, if n is not divisible by small primes, and p is small, n likely has a large factor.
            # Let's stick to small primes first.
            # If n > 1 after small primes, and p <= 10^6:
            # We need to check if n has any prime factor > p.
            # If n is prime and n > p, return False.
            # If n is composite, it might have factors <= p and > p.
            
            # Fast path: if n is prime
            if is_prime_mr(n):
                return n <= p
            
            # If n is composite, try to factor it.
            # If p is small, we can trial divide up to p.
            # But that's slow.
            # Better: factor n completely and check max factor.
            
            factors = []
            stack = [n]
            while stack:
                curr = stack.pop()
                if curr == 1: continue
                if is_prime_mr(curr):
                    factors.append(curr)
                    continue
                f = pollard_brent(curr)
                if f == curr:
                    factors.append(curr)
                else:
                    stack.append(f)
                    stack.append(curr // f)
            
            for f in factors:
                if f > p:
                    return False
            return True
        
        else:
            # p > 10^6
            # We divided by small primes.
            # n is now either 1, prime, or composite with factors > 1223.
            # Any remaining factor must be <= p.
            
            if n == 1:
                return True
            
            if is_prime_mr(n):
                return n <= p
            
            # Factor n
            factors = []
            stack = [n]
            while stack:
                curr = stack.pop()
                if curr == 1: continue
                if is_prime_mr(curr):
                    factors.append(curr)
                    continue
                f = pollard_brent(curr)
                if f == curr:
                    factors.append(curr)
                else:
                    stack.append(f)
                    stack.append(curr // f)
            
            for f in factors:
                if f > p:
                    return False
            return True

    def check_run(p, primes_list):
        p_sq = p * p
        k = 0
        for j in range(1, 201):
            n = p_sq + j
            if is_p_smooth(n, p):
                k = j
            else:
                break
        return k

    best_k = 0
    best_p = 2

    # 2. Exhaustive search for p <= 10^6
    for p in primes:
        if time.time() > deadline:
            break
        k = check_run(p, primes)
        if k > best_k:
            best_k = k
            best_p = p
            if best_k >= 100:
                break

    # 3. Search for larger primes using segmented sieve
    # This is the structural change: use segmented sieve to find candidate primes in large ranges
    # and check them.
    
    # Segmented sieve parameters
    SEG_SIZE = 10**7
    LIMIT = 10**9
    
    # We need small primes for segmentation
    # We already have primes up to 10^6. We need primes up to sqrt(10^9) = 31623.
    # So primes list is sufficient.
    
    # Search from high to low? Or random segments?
    # High p means p^2 is large, factors are large.
    # Let's try random segments to find a good p quickly.
    
    random.seed(42)
    checked_primes = set()
    
    while time.time() < deadline - 5:
        # Pick a random start for a segment
        start = random.randint(10**6, LIMIT - SEG_SIZE)
        start = (start // 2) * 2 + 1 # Make it odd
        
        # Segmented sieve for [start, start + SEG_SIZE]
        seg = [True] * SEG_SIZE
        
        for p in primes:
            if p * p > start + SEG_SIZE:
                break
            # Find first multiple of p in [start, start + SEG_SIZE]
            first = ((start + p - 1) // p) * p
            if first == p:
                first += p
            for j in range(first - start, SEG_SIZE, p):
                seg[j] = False
        
        # Collect primes in this segment
        for i in range(0, SEG_SIZE, 2): # Only check odd numbers
            if seg[i]:
                p = start + i
                if p <= LIMIT and p not in checked_primes:
                    checked_primes.add(p)
                    if time.time() > deadline - 5:
                        break
                    k = check_run(p, primes)
                    if k > best_k:
                        best_k = k
                        best_p = p
                        if best_k >= 100:
                            return {'p': best_p, 'k': best_k}

    return {'p': best_p, 'k': best_k}
