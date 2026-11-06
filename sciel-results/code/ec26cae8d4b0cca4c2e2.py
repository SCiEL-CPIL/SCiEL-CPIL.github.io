"""
Strategy: 
1. Use deterministic Miller-Rabin for fast primality testing.
2. Use Pollard-Brent for fast factorization of large integers.
3. Generate candidate primes using a segmented sieve up to 10^9.
4. Search primes in descending order (larger p -> higher chance of smoothness).
5. For each prime, check p^2+1 ... p^2+k for largest prime factor <= p.
6. Stop when time budget is nearly exhausted, returning the best found.
"""

import time
import random
import math
import sys

# Constants
TIME_LIMIT = 120  # seconds
SAFE_MARGIN = 5   # seconds to leave before hard kill
MAX_P = 10**9
MAX_K = 200

def is_prime(n):
    if n < 2:
        return False
    # Small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    # Miller-Rabin
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # Bases sufficient for n < 3,474,749,660,383
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
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
    while True:
        y = random.randrange(1, n)
        c = random.randrange(1, n)
        m = random.randrange(1, n)
        g = r = q = 1
        while g == 1:
            x = y
            for i in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for i in range(min(m, r - k)):
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

def factorize(n):
    """Returns a list of prime factors of n."""
    factors = []
    # Trial division for small primes to speed up
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
        while n % p == 0:
            factors.append(p)
            n //= p
    if n == 1:
        return factors
    stack = [n]
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if is_prime(m):
            factors.append(m)
        else:
            d = pollard_brent(m)
            stack.append(d)
            stack.append(m // d)
    return factors

def largest_prime_factor(n):
    if n <= 1:
        return 1
    factors = factorize(n)
    return max(factors)

def segmented_sieve(limit):
    """
    Yields primes up to limit. 
    Since we need primes up to 10^9, we can't sieve the whole range in memory easily 
    without optimization, but 10^9 is manageable with a segmented approach or 
    we can just test random numbers. 
    However, the task asks to maximize k. Larger p is generally better. 
    Let's try a random search strategy first as it's simpler and often effective 
    for this specific "smoothness" property, or a segmented sieve from high to low.
    
    Actually, a full sieve up to 10^9 is ~125MB of bits, which is fine. 
    But generating 50 million primes is slow. 
    Random search is better: pick random large primes, check them.
    """
    pass

def construct():
    start_time = time.time()
    best_k = 1
    best_p = 2
    
    # Pre-compute small primes for trial division in factorize
    # We already have small primes in is_prime, let's reuse or define locally
    
    # Strategy: Random search for large primes
    # Why random? The condition "all factors <= p" for p^2+i is related to 
    # p-smooth numbers. The probability that a random number N is p-smooth 
    # is roughly rho(u) where u = log(N)/log(p). 
    # Here N ~ p^2, so log(N) ~ 2 log(p), u ~ 2. 
    # rho(2) is small (~0.3). 
    # So we expect to find such runs. 
    # Larger p might not necessarily be better if the density doesn't change much, 
    # but we are limited to k<=200. 
    # Let's search in chunks of primes.
    
    # To be safe and fast, we'll test primes in a specific range.
    # Let's try primes near 10^9, then 10^8, etc.
    
    # Helper to check a specific prime
    def check_prime(p, max_k_to_check):
        if not is_prime(p):
            return 0
        k = 0
        p_sq = p * p
        for i in range(1, max_k_to_check + 1):
            n = p_sq + i
            # Quick check: if n is prime, it must be <= p. 
            # But n > p^2 >= p for p>=2. So if n is prime, it fails.
            # Wait, if n is prime, its largest factor is n. 
            # Condition: largest factor <= p.
            # Since n = p^2 + i > p (for p>1, i>=1), if n is prime, largest factor is n > p.
            # So n MUST be composite and all its prime factors must be <= p.
            
            # Optimization: 
            # If n is prime, fail immediately.
            if is_prime(n):
                break
            
            # Factorize n
            # We only care if any factor > p.
            # Pollard-Brent is fast, but we can add a check:
            # If n has a factor d such that d > p, fail.
            # Actually, factorize and check max is robust.
            
            try:
                lpf = largest_prime_factor(n)
                if lpf > p:
                    break
            except Exception:
                # In case of random seed issues or infinite loops in Pollard
                break
            
            k += 1
            
            # Time check inside loop
            if time.time() - start_time > TIME_LIMIT - SAFE_MARGIN:
                return k
        
        return k

    # Search Strategy:
    # 1. Try a few known "good" primes if we had them (we don't).
    # 2. Random search in high ranges.
    # 3. Segmented sieve in descending order? 
    #    Sieving 10^9 down is hard. 
    #    Let's do random sampling of primes.
    
    # To ensure we find *some* prime, we start with small primes if random fails? 
    # No, small primes have low k. 
    # We want to maximize k. 
    # Let's sample random primes from [10^8, 10^9], [10^7, 10^8], etc.
    
    # We need a fast way to get a random prime.
    # Pick random odd number, check primality.
    
    def get_random_prime(min_val, max_val):
        while True:
            n = random.randrange(min_val, max_val, 2)
            if n < 2:
                n = 2
            if is_prime(n):
                return n

    # Ranges to search
    # We prioritize larger primes.
    ranges = [
        (10**8, 10**9),
        (5*10**7, 10**8),
        (10**7, 5*10**7),
        (10**6, 10**7),
        (10**5, 10**6),
        (10**4, 10**5),
        (1000, 10**4),
        (100, 1000),
        (10, 100),
        (2, 10)
    ]
    
    # We will iterate through these ranges.
    # For each range, we'll try a number of random primes.
    # The number of tries should depend on time remaining.
    
    # First, do a quick scan of small primes to establish a baseline? 
    # The seed program found k=2. Let's just go for high primes.
    
    for min_p, max_p in ranges:
        # Estimate how many primes to try
        # Time left
        elapsed = time.time() - start_time
        remaining = TIME_LIMIT - SAFE_MARGIN - elapsed
        if remaining < 1:
            break
        
        # Try random primes in this range
        # How many? 
        # Each check takes roughly 10-100ms for large p (factorization of 200 numbers).
        # Let's try 50 primes per range initially, adjusting if fast.
        
        # Actually, let's just loop until time runs out for the first few ranges,
        # then move to smaller if we want to be safe, but larger is better.
        
        # Let's try 100 random primes in the highest range first.
        attempts = 0
        max_attempts = 1000 # Cap per range to avoid getting stuck in one range if it's bad
        
        while attempts < max_attempts:
            if time.time() - start_time > TIME_LIMIT - SAFE_MARGIN:
                break
            
            p = get_random_prime(min_p, max_p)
            # Check this prime
            # We check up to MAX_K
            k_found = check_prime(p, MAX_K)
            
            if k_found > best_k:
                best_k = k_found
                best_p = p
                # If we found a high k, we might want to try more primes of similar size
                # to see if we can get even higher (up to 200).
                # But if we hit 200, we can stop immediately.
                if best_k >= MAX_K:
                    break
            attempts += 1
            
            # If we found a decent k, we can reduce attempts for this range 
            # and move to next? No, stick with this range if it's producing good results.
            # If we found k=0 or 1 repeatedly, move to next range.
            # Simple heuristic: if best_k hasn't improved in last 20 tries, break range.
            # (Implemented simply: just let it run, time limit will stop it).
            
            # Optimization: if we are in a high range and finding small k, 
            # maybe switch to a slightly lower range? 
            # For now, just random within range.
            
        if best_k >= MAX_K:
            break

    # Final check: ensure we return a valid prime
    # If best_p is 2, we should check if we actually verified it.
    # check_prime(2, ...) -> 2^2+1=5 (prime > 2) -> k=0.
    # So best_k would be 0. 
    # The task requires k >= 1.
    # If we didn't find any k>=1, we must find one.
    # Small primes might have k=1? 
    # p=3: 9+1=10 (2,5). 5>3. Fail. k=0.
    # p=5: 25+1=26 (2,13). 13>5. Fail.
    # p=7: 49+1=50 (2,5). 5<=7. OK. 49+2=51 (3,17). 17>7. Fail. k=1.
    # So p=7 gives k=1.
    # If our random search failed to find anything (unlikely), we default to p=7, k=1.
    
    if best_k < 1:
        best_p = 7
        best_k = 1
        # Verify p=7
        # 49+1=50=2*5*5. Max factor 5 <= 7.
        # 49+2=51=3*17. Max factor 17 > 7.
        # So k=1 is correct.

    return {'p': best_p, 'k': best_k}

# Main entry point
if __name__ == "__main__":
    # Just for local testing
    result = construct()
    print(result)
