import math
import time
import random
import sys

# Deterministic Miller-Rabin bases for n < 2^64
MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in MR_BASES:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
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
        y = random.randrange(2, n)
        c = random.randrange(1, n)
        m = 128  # Fixed m for better performance in Brent's cycle detection
        g = 1
        r = 1
        q = 1
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

def is_p_smooth(n, p):
    """
    Returns True if all prime factors of n are <= p.
    """
    if n == 1:
        return True
    
    # If n is prime, check directly
    if is_prime(n):
        return n <= p
    
    # Find a factor
    d = pollard_brent(n)
    if d == n:
        # Should not happen for composite, but handle edge case
        return is_p_smooth(n, p)
    
    return is_p_smooth(d, p) and is_p_smooth(n // d, p)

def check_run(p, max_k=200, deadline=None):
    """
    Check how long the run is for prime p.
    Returns k such that p^2+1 ... p^2+k are all p-smooth.
    """
    k = 0
    start_val = p * p + 1
    for i in range(1, max_k + 1):
        if deadline is not None and time.time() > deadline:
            break
        val = start_val + i - 1
        if not is_p_smooth(val, p):
            return k
        k += 1
    return k

def sieve(n):
    """
    Sieve of Eratosthenes.
    Returns list of primes up to n.
    """
    is_prime = bytearray(b'\x01') * (n + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:n+1:step] = bytearray(len(is_prime[start:n+1:step]))
    return [i for i, prime in enumerate(is_prime) if prime]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    # Precompute small primes for the heuristic filter
    # We use primes up to 100 for the "smoothness score"
    small_primes_for_filter = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    # We will search in two phases:
    # Phase 1: Sieve up to 200,000 (fast)
    # Phase 2: If time remains, sieve up to 1,000,000 (slower but covers more primes)
    
    best_k = 0
    best_p = 2
    
    # Phase 1: Small primes
    limit_1 = 200000
    primes_1 = sieve(limit_1)
    
    # Heuristic: Score a prime p based on how many of the first 20 numbers (p^2+1 ... p^2+20)
    # are divisible by at least one small prime from our filter set.
    # A higher score suggests a higher likelihood of being p-smooth (since small factors are more likely).
    # Note: This is a proxy. We want to find primes where p^2+i has small factors.
    # If p^2+i is prime, it has no small factors (unless it IS a small factor, but p^2+i > p for i>=1, so it can't be <= p).
    # Actually, if p^2+i is composite, it likely has small factors.
    # So we want to maximize the count of "has small factor" in the window.
    
    # However, a simpler and more direct heuristic for "p-smoothness" is:
    # For each i in 1..K, check if p^2+i has any factor < some threshold T.
    # If it has a factor < T, it's "more likely" to be p-smooth (assuming T <= p).
    # If it has no factor < T, it might be prime or have large factors.
    
    # Let's use a window of 10 for the heuristic to be fast.
    WINDOW = 10
    
    def score_prime(p):
        score = 0
        p2 = p * p
        for i in range(1, WINDOW + 1):
            val = p2 + i
            for sp in small_primes_for_filter:
                if val % sp == 0:
                    score += 1
                    break # Count each val only once
        return score
    
    # We will test all primes in Phase 1, but we can sort them by score to prioritize?
    # Sorting 17k primes is cheap.
    # But calculating scores for 17k primes * 10 windows * 25 primes = ~4M ops. Very fast.
    
    scored_primes = []
    for p in primes_1:
        if time.time() > deadline - 5: # Keep 5s buffer
            break
        s = score_prime(p)
        scored_primes.append((s, p))
        
    # Sort by score descending
    scored_primes.sort(reverse=True)
    
    # Verify top candidates
    # How many to verify?
    # If we verify all, it's the same as before.
    # Let's verify the top 1000 first.
    # If we find a good k, we might stop early or continue.
    
    verified_count = 0
    for s, p in scored_primes:
        if time.time() > deadline - 5:
            break
        
        # Only verify if score is "good enough"? 
        # Or just verify top N.
        # Let's verify top 2000 from this batch.
        if verified_count < 2000:
             k = check_run(p, 200, deadline)
             if k > best_k:
                 best_k = k
                 best_p = p
                 if best_k == 200:
                     return {'p': best_p, 'k': best_k}
             verified_count += 1
        else:
             # If we've verified 2000, we can break if time is tight, or continue if fast?
             # check_run is the bottleneck.
             # Let's break to save time for Phase 2.
             break
             
    # Phase 2: Larger primes
    if time.time() < deadline - 10:
        limit_2 = 1000000
        # We need primes between limit_1 and limit_2
        # Sieving again is easiest, or segmented. Let's just sieve again and filter.
        # Sieve 1M is ~0.1-0.2s in Python with bytearray.
        primes_2 = sieve(limit_2)
        # Filter out primes <= limit_1
        # To avoid re-scoring, we can just iterate primes_2 and skip small ones.
        
        # We don't have the scores for these. We need to calculate them.
        # But we can be more selective.
        # Let's calculate scores and pick top 500.
        
        scored_primes_2 = []
        for p in primes_2:
            if p <= limit_1:
                continue
            if time.time() > deadline - 10:
                break
            s = score_prime(p)
            scored_primes_2.append((s, p))
            
        if scored_primes_2:
            scored_primes_2.sort(reverse=True)
            # Verify top 500
            for s, p in scored_primes_2[:500]:
                if time.time() > deadline - 2:
                    break
                k = check_run(p, 200, deadline)
                if k > best_k:
                    best_k = k
                    best_p = p
                    if best_k == 200:
                        return {'p': best_p, 'k': best_k}

    return {'p': best_p, 'k': best_k}

if __name__ == "__main__":
    res = construct()
    print(res)
