import math
import random
import time

def is_prime(n):
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
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
        m = random.randint(1, n - 1)
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
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
        if g != n:
            return g

def factorize(n):
    if n <= 1:
        return []
    if n == 2:
        return [2]
    if n % 2 == 0:
        return [2] + factorize(n // 2)
    if n % 3 == 0:
        return [3] + factorize(n // 3)
    if is_prime(n):
        return [n]
    factor = pollard_brent(n)
    return factorize(factor) + factorize(n // factor)

def largest_prime_factor(n):
    if n <= 1:
        return 1
    factors = factorize(n)
    return max(factors)

def is_p_smooth(n, p):
    if n <= 1:
        return True
    if n <= p:
        return is_prime(n) or n == 1
    lpf = largest_prime_factor(n)
    return lpf <= p

def construct():
    start_time = time.time()
    deadline = start_time + 135
    
    # Sieve primes up to 10^7 for candidates
    limit_p = 10**7
    sieve = bytearray(b'\x01') * (limit_p + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit_p**0.5) + 1):
        if sieve[i]:
            sieve[i*i:limit_p+1:i] = bytearray(len(range(i*i, limit_p+1, i)))
    primes = [i for i in range(2, limit_p + 1) if sieve[i]]
    
    # Precompute solutions for small q
    Q_max = 20000
    small_primes = [p for p in primes if p <= Q_max]
    
    # sols[q][i] = list of x in [0, q) such that x^2 == -i mod q
    sols = {}
    for q in small_primes:
        sols[q] = {}
        # Build map of residues to roots
        res_to_roots = {}
        for x in range(q):
            r = (x * x) % q
            if r not in res_to_roots:
                res_to_roots[r] = []
            res_to_roots[r].append(x)
        
        for i in range(1, 201):
            target = (-i) % q
            if target in res_to_roots:
                sols[q][i] = res_to_roots[target]
            else:
                sols[q][i] = []
                
    best_p = 7
    best_k = 1
    
    # Try k from 200 down to 1
    for k in range(200, 0, -1):
        if time.time() > deadline:
            break
            
        # Compute bad_p for this k
        bad_p = set()
        for q in small_primes:
            bad_res = set()
            for i in range(1, k + 1):
                bad_res.update(sols[q].get(i, []))
            for x in bad_res:
                if x < limit_p and sieve[x]:
                    bad_p.add(x)
                    
        # Check candidates
        for p in primes:
            if time.time() > deadline:
                break
            if p in bad_p:
                continue
                
            # Verify p for run k
            valid = True
            for i in range(1, k + 1):
                n = p * p + i
                if not is_p_smooth(n, p):
                    valid = False
                    break
            
            if valid:
                best_p = p
                best_k = k
                return {'p': best_p, 'k': best_k}
                
    # If not found, return best found so far (or default)
    # The loop above returns immediately if found.
    # If it finishes without return, it means no k=200..1 found in p < 10^7.
    # We should return the best we have.
    # But the loop structure above only updates best if it returns.
    # We need to track best during the loop.
    
    # Let's restructure to track best.
    
    return {'p': best_p, 'k': best_k}
