import time
import numpy as np

def construct():
    """
    Finds n in [10000, 1000000] that minimizes max(LPF(n), LPF(n+1)).
    Uses a segmented sieve with NumPy for efficient LPF computation.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    LO = 10000
    HI = 1000000
    MAX_VAL = HI + 1
    
    # Step 1: Find all primes up to sqrt(MAX_VAL) using simple sieve
    limit = int(np.sqrt(MAX_VAL)) + 1
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    primes = np.where(is_prime)[0]
    
    # Step 2: Use segmented sieve to compute LPF for [LO, MAX_VAL]
    # We only need LPF for indices LO to MAX_VAL
    size = MAX_VAL - LO + 1
    lpf_segment = np.zeros(size, dtype=np.int32)
    
    # For each prime, find multiples in [LO, MAX_VAL] and update LPF
    for p in primes:
        # First multiple of p >= LO
        first = ((LO + p - 1) // p) * p
        # Mark all multiples of p in the segment
        idx = (first - LO)
        while idx < size:
            lpf_segment[idx] = p  # Since we iterate primes in order, last write wins
            idx += p
    
    # Also handle the case where a number is prime itself (greater than sqrt(MAX_VAL))
    # For numbers in [LO, MAX_VAL] that are prime, their LPF is themselves
    # We need to check which numbers in the segment are prime
    # A number n is prime if it has no prime factor <= sqrt(n)
    # Since we've marked all composites with their largest prime factor <= sqrt(MAX_VAL),
    # numbers with lpf_segment == 0 are either 1 or primes > sqrt(MAX_VAL)
    
    # For numbers where lpf_segment is 0, they could be primes (or 1, but LO >= 10000)
    # Actually, if n > 1 and all its prime factors are > sqrt(MAX_VAL), then n must be prime
    # because if n = a*b with a <= b, then a <= sqrt(n) <= sqrt(MAX_VAL)
    # So if no prime <= sqrt(MAX_VAL) divides n, then n is prime
    zero_mask = (lpf_segment == 0)
    if np.any(zero_mask):
        # These numbers are primes (since LO >= 10000 > 1)
        indices = np.where(zero_mask)[0]
        actual_values = LO + indices
        lpf_segment[zero_mask] = actual_values
    
    # Step 3: Compute max(LPF(n), LPF(n+1)) for n in [LO, HI]
    # lpf_segment[i] corresponds to value LO + i
    # We need max(lpf_segment[n-LO], lpf_segment[n+1-LO]) for n in [LO, HI]
    
    # Create arrays for n and n+1
    n_vals = np.arange(LO, HI + 1)
    idx_n = n_vals - LO  # indices into lpf_segment for n
    idx_np1 = (n_vals + 1) - LO  # indices into lpf_segment for n+1
    
    lpf_n = lpf_segment[idx_n]
    lpf_np1 = lpf_segment[idx_np1]
    
    # Compute the max for each n
    max_lpf = np.maximum(lpf_n, lpf_np1)
    
    # Find the minimum
    min_idx = np.argmin(max_lpf)
    best_n = n_vals[min_idx]
    best_val = max_lpf[min_idx]
    
    # Check time - if we're close to deadline, just return
    if time.time() > deadline:
        return best_n
    
    return int(best_n)
