import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin within 90s limit

    # We need largest prime factor for values up to 1,000,001 (n+1 when n=1,000,000)
    MAXV = 1000001

    # Build largest-prime-factor array using a sieve.
    # lpf[i] will hold the largest prime factor of i.
    lpf = np.zeros(MAXV + 1, dtype=np.int64)
    
    # First, find all primes using a simple sieve
    is_prime = np.ones(MAXV + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(np.sqrt(MAXV)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
            
    # Get list of primes
    primes = np.where(is_prime)[0]
    
    # Now mark LPF for each prime
    for p in primes:
        lpf[p::p] = p

    # Now find n in [10000, 1000000] minimizing max(lpf[n], lpf[n+1])
    # Use vectorized numpy for speed
    n_arr = np.arange(10000, 1000001, dtype=np.int64)
    lpf_n = lpf[n_arr]
    lpf_np1 = lpf[n_arr + 1]
    max_lpf = np.maximum(lpf_n, lpf_np1)

    # Find the index of the minimum
    min_idx = np.argmin(max_lpf)
    best_n = int(n_arr[min_idx])
    best_m = int(max_lpf[min_idx])

    # The answer is deterministic; no need for iterative improvement
    # Verify we're within time budget
    if time.time() > deadline:
        pass  # still return best found

    return best_n
