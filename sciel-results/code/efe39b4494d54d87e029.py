"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Find n in [10000, 1000000] that minimizes the largest prime factor of n(n+1).
    
    Strategy:
    1. Precompute the smallest prime factor (SPF) for all numbers up to 1,000,001
       using a linear sieve.
    2. For each n in [10000, 1000000], factorize n and n+1 using SPF to get
       their largest prime factors in O(log n) time.
    3. Track the n with the minimum max(LPF[n], LPF[n+1]).
    4. Return that n.
    
    Deadline guard: If precomputation or scan takes too long, return best found so far.
    """
    start_time = time.time()
    deadline = start_time + 80  # 10 second safety margin before 90s kill
    
    MAX_VAL = 1000001
    
    # Linear sieve to compute smallest prime factor
    spf = list(range(MAX_VAL + 1))
    for i in range(2, int(MAX_VAL**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, MAX_VAL + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    if time.time() > deadline:
        return 10000
    
    # Helper to compute largest prime factor using SPF
    def largest_prime_factor(x):
        if x <= 1:
            return 1
        max_p = 1
        while x > 1:
            p = spf[x]
            max_p = p
            while x % p == 0:
                x //= p
        return max_p
    
    # Now scan for best n
    best_n = 10000
    best_lpf_val = float('inf')
    
    # n ranges from 10000 to 1000000
    # We need LPF of n and n+1
    # n+1 goes up to 1000001, which is within MAX_VAL
    
    for n in range(10000, 1000001):
        lpf_n = largest_prime_factor(n)
        lpf_np1 = largest_prime_factor(n + 1)
        current_max_lpf = lpf_n if lpf_n > lpf_np1 else lpf_np1
        if current_max_lpf < best_lpf_val:
            best_lpf_val = current_max_lpf
            best_n = n
        if time.time() > deadline:
            break
    
    return best_n

# EVOLVE-BLOCK-END
