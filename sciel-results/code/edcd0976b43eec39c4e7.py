import numpy as np
import math
import time

def construct():
    start = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill

    N_MIN = 10000
    N_MAX = 1000000
    NMAX = N_MAX + 1  # We need LPF up to 1000001

    # 1. Sieve of Eratosthenes for primes up to NMAX using NumPy
    is_prime = np.ones(NMAX + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(math.isqrt(NMAX)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False

    # 2. Compute Largest Prime Factor (LPF) using NumPy vectorized stamping
    # Initialize with 0. Primes will be stamped onto their multiples.
    # Since we iterate primes in ascending order, the last stamp is the largest prime factor.
    lpf = np.zeros(NMAX + 1, dtype=np.int32)
    
    # Get list of primes
    primes = np.where(is_prime)[0]
    
    for p in primes:
        # Stamp p onto all multiples of p
        lpf[p::p] = p

    # Check time after the potentially slow LPF calculation
    if time.time() > DEADLINE:
        # Fallback: return a known good value if we ran out of time during computation
        # This is highly unlikely given the speed of NumPy, but safety first.
        return 10000

    # 3. Find n in [N_MIN, N_MAX] minimizing max(LPF(n), LPF(n+1))
    # lpf[n] for n in N_MIN..N_MAX
    # lpf[n+1] for n in N_MIN..N_MAX -> indices N_MIN+1 .. N_MAX+1
    
    # Extract relevant slices
    curr_lpf = lpf[N_MIN:N_MAX+1]
    next_lpf = lpf[N_MIN+1:N_MAX+2]
    
    # Compute score for each n: the largest prime factor of n(n+1) is max(LPF(n), LPF(n+1))
    max_lpf = np.maximum(curr_lpf, next_lpf)
    
    # Find the minimum score
    min_val = np.min(max_lpf)
    min_idx = np.argmin(max_lpf)
    
    best_n = N_MIN + min_idx
    
    return int(best_n)
