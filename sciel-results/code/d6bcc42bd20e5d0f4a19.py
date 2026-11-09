import time
import math
import random
import sys
import heapq

# ---------- Deterministic Miller-Rabin for 64-bit ----------
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def _is_prime_mr(n):
    if n < 2:
        return False
    for p in _MR_BASES:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_BASES:
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

# ---------- Pollard-Brent ----------
def _pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randrange(1, n)
        c = random.randrange(1, n)
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
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g != 1:
                    break
        if g != n:
            return g

def _factor(n, result):
    if n <= 1:
        return
    if _is_prime_mr(n):
        result.append(n)
        return
    d = _pollard_brent(n)
    _factor(d, result)
    _factor(n // d, result)

# ---------- Sieve ----------
def _sieve(limit):
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin

    # Precompute small primes for fast scoring
    SIEVE_LIMIT = 1000000
    all_primes = _sieve(SIEVE_LIMIT)
    
    # Use first 200 primes for fast scoring
    SCORE_PRIMES = all_primes[:200]
    
    best_k = 0
    best_p = 3
    
    # Candidate list: (score, p)
    # We want to keep the top candidates
    candidates = []
    MAX_CANDIDATES = 500
    
    # Phase 1: Fast Scoring
    # Iterate through primes
    # We can go up to SIEVE_LIMIT
    for p in all_primes:
        if time.time() > deadline - 20.0: # Leave time for verification
            break
            
        p_sq = p * p
        score = 0
        
        # Fast check: count small factors
        for i in range(1, 201):
            n = p_sq + i
            # Check divisibility by SCORE_PRIMES
            # If n becomes 1, it's fully smooth by small primes
            # If n > 1, check if it's "likely" smooth
            
            # Optimization: If n is even, divide by 2
            # We just count how many small primes divide it
            # And if the remainder is small
            
            t = n
            small_factor_count = 0
            for q in SCORE_PRIMES:
                if q > p:
                    break
                if q * q > t:
                    break
                if t % q == 0:
                    small_factor_count += 1
                    while t % q == 0:
                        t //= q
            if t == 1:
                score += 10
            elif t <= p:
                # t is prime or composite with factors <= p (since we divided by small primes, t might be prime > p? No, if t <= p, it's fine)
                # Wait, if t <= p, it's p-smooth.
                score += 5
            else:
                # t > p. It has a factor > p?
                # Not necessarily. t could be product of two primes > p?
                # If t > p and we've divided by all primes <= min(p, sqrt(t)), then t is prime or product of primes > p.
                # If t is prime > p, it fails.
                # If t is composite, it has a factor > p.
                # So if t > p, it likely fails.
                score -= 2
                # Early break for this i if score is already low?
                # No, we need to sum over all i.
                
        # Add to candidates if score is good
        # We want to keep the top MAX_CANDIDATES
        heapq.heappush(candidates, (-score, p))
        if len(candidates) > MAX_CANDIDATES:
            heapq.heappop(candidates)

    # Phase 2: Verification
    # Sort candidates by score (descending)
    candidates = [(-s, p) for s, p in candidates]
    candidates.sort()
    
    for score, p in candidates:
        if time.time() > deadline:
            break
            
        p_sq = p * p
        k = 0
        for i in range(1, 201):
            val = p_sq + i
            # Full smoothness check
            t = val
            is_smooth = True
            for q in all_primes:
                if q > p:
                    break
                if q * q > t:
                    break
                while t % q == 0:
                    t //= q
            if t != 1:
                # t > 1. Check if t is prime and <= p
                if _is_prime_mr(t):
                    if t <= p:
                        is_smooth = True
                    else:
                        is_smooth = False
                else:
                    # t is composite. It has a factor > p?
                    # Since we divided by all primes <= p, any factor of t must be > p.
                    # So it fails.
                    is_smooth = False
            
            if is_smooth:
                k = i
            else:
                break
                
        if k > best_k:
            best_k = k
            best_p = p
            if best_k >= 200:
                return {'p': best_p, 'k': best_k}
                
    return {'p': best_p, 'k': best_k}
