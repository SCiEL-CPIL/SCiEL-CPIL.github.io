"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def _largest_prime_factor(n):
    if n <= 1:
        return 1
    lpf = 1
    # Trial divide by 2
    while n % 2 == 0:
        lpf = 2
        n //= 2
    # Trial divide by odd numbers up to sqrt(n)
    d = 3
    while d * d <= n:
        while n % d == 0:
            lpf = d
            n //= d
        d += 2
    if n > 1:
        lpf = n
    return lpf

def _sieve_primes(limit):
    if limit < 2:
        return []
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i*i, limit+1, i):
                is_prime[j] = False
    return [i for i, v in enumerate(is_prime) if v]

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline to be safe within 90s
    
    # Strategy: For each candidate n in [10000, 1000000], compute L(n) = max(LPF(n), LPF(n+1))
    # We want to minimize L(n).
    # A naive scan over 1 million numbers with trial division might be slow in Python.
    # Let's estimate: 10^6 numbers. Average LPF computation for n up to 10^6.
    # sqrt(10^6) = 1000. So trial division up to 1000.
    # 10^6 * 1000 operations = 10^9 operations. This is likely too slow for pure Python (5-10s per 10^8 ops -> 10-100s).
    # We need a smarter approach or optimization.
    
    # Optimization 1: Precompute smallest prime factor (SPF) or just primes up to sqrt(1000000) = 1000.
    # Actually, we can precompute the largest prime factor for all numbers up to 1000001 using a sieve-like method.
    # Let's use a sieve to compute LPF for all numbers in range.
    # This is O(N log log N) or similar.
    # N = 1000001.
    # This is very fast in Python if done carefully, but list operations might be slow.
    # Let's try a sieve for largest prime factor.
    
    max_n = 1000000
    max_val = max_n + 1
    
    # Sieve for largest prime factor
    lpf = [0] * (max_val + 1)
    for i in range(2, max_val + 1):
        if lpf[i] == 0: # i is prime
            for j in range(i, max_val + 1, i):
                lpf[j] = i # i is the largest prime factor of j so far (since we go increasing)
                # Wait, if we iterate i increasing, and set lpf[j] = i, then for composite j,
                # the last prime i that divides j will be the largest.
                # Example: j=12. i=2 -> lpf[12]=2. i=3 -> lpf[12]=3. i=4 (not prime, skipped).
                # So yes, this works.
                
    # Now we have lpf[i] for all i in [0, max_val]
    # We want to find n in [10000, 1000000] that minimizes max(lpf[n], lpf[n+1])
    
    best_n = 10000
    best_val = float('inf')
    
    for n in range(10000, max_n + 1):
        val = max(lpf[n], lpf[n+1])
        if val < best_val:
            best_val = val
            best_n = n
        # Early termination check? Not really, but we can break if we find a very small value?
        # The minimum possible value is 2 (if both are powers of 2? No, consecutive integers are coprime.
        # One is even, one is odd.
        # If n is a power of 2, lpf[n]=2. n+1 is odd.
        # Can n+1 have lpf 2? No, it's odd.
        # So min max(lpf[n], lpf[n+1]) is at least 3?
        # If n=2^k, n+1 is odd. Smallest odd prime is 3.
        # So if n+1 is a power of 3, lpf[n+1]=3.
        # Then max(2, 3) = 3.
        # Are there consecutive integers where one is 2-power and other is 3-power?
        # 2^a = 3^b + 1.
        # 2^2 = 4, 4-1=3 (3^1). So n=4, n+1=3. But n must be >= 10000.
        # 2^a - 3^b = 1.
        # Catalan's conjecture (Mihăilescu's theorem) says only 3^2 - 2^3 = 1.
        # So no other solutions.
        # Thus, for n >= 10000, we can't have lpf=3 for both?
        # Wait, we need max(lpf[n], lpf[n+1]) to be small.
        # If we find max=3, that's the global minimum possible for n>2.
        # Let's check if any n in range has max(lpf[n], lpf[n+1]) == 3.
        # This requires {lpf[n], lpf[n+1]} subset of {2, 3}.
        # So n and n+1 are of the form 2^a * 3^b.
        # Consecutive integers both 3-smooth.
        # Known pairs: (2,3), (3,4), (4,5-no), (8,9), (9,10-no), (16,17-no)...
        # (8,9): 8=2^3, 9=3^2. max(2,3)=3.
        # Next pair of consecutive 3-smooth numbers?
        # They are sparse.
        # Let's just run the sieve and find the min.
        
        if time.time() > deadline:
            break
            
    return best_n
# EVOLVE-BLOCK-END
