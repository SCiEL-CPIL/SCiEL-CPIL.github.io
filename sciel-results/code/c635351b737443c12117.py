"""Rewritten program for the suite task; uses y-smooth bitmask approach."""
import time
import math

def construct():
    """
    Find n in [10000, 1000000] that minimizes the largest prime factor of n(n+1).
    
    Strategy:
    1. Precompute the smallest prime factor (SPF) for all numbers up to 1,000,001.
    2. For each candidate n in [10000, 1000000], compute the largest prime factor
       of n and n+1 using the SPF table, then take the max.
    3. Track the n with the minimum such value.
    4. Return that n.
    
    Deadline guard: If computation takes too long, return best found so far.
    """
    start_time = time.time()
    deadline = start_time + 55  # 5 second safety margin before 60s limit
    
    # We need prime factors for numbers up to 1,000,001
    MAX_VAL = 1000001
    
    # Sieve to compute smallest prime factor
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(math.isqrt(MAX_VAL)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    def largest_prime_factor(x):
        """Compute largest prime factor of x using SPF table."""
        if x <= 1:
            return 1
        max_p = 1
        while x > 1:
            p = spf[x]
            max_p = max(max_p, p)
            while x % p == 0:
                x //= p
        return max_p
    
    # Scan for best n
    best_n = 10000
    best_lpf_val = float('inf')
    
    for n in range(10000, 1000001):
        lpf_n = largest_prime_factor(n)
        lpf_np1 = largest_prime_factor(n + 1)
        current_max_lpf = max(lpf_n, lpf_np1)
        if current_max_lpf < best_lpf_val:
            best_lpf_val = current_max_lpf
            best_n = n
        # Check time periodically
        if n % 50000 == 0 and time.time() > deadline:
            break
    
    return best_n
