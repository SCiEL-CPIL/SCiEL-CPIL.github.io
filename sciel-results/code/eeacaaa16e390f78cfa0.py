"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Find integer n in [10000, 1000000] that minimizes the largest prime factor of n(n+1).
    
    Strategy:
    1. Compute the largest prime factor (LPF) for every integer up to 1,000,001 using an efficient sieve.
    2. For each n in [10000, 1000000], compute LPF(n * (n+1)) = max(LPF(n), LPF(n+1)).
    3. Find the n that minimizes this value using vectorized NumPy operations.
    
    We use an array `lpf` where lpf[i] is the largest prime factor of i.
    """
    MAX_N = 1000000
    MAX_VAL = MAX_N + 1  # We need LPF up to 1000001
    
    # Initialize LPF array with 0
    lpf = np.zeros(MAX_VAL + 1, dtype=np.int32)
    
    # Efficient sieve for largest prime factor:
    # For each prime p, update all multiples of p.
    # Since we process primes in increasing order, the last prime that updates a number
    # will be its largest prime factor.
    
    # First, identify primes using a simple sieve
    is_prime = np.ones(MAX_VAL + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    
    for i in range(2, int(np.sqrt(MAX_VAL)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False
    
    # Get list of primes
    primes = np.where(is_prime)[0]
    
    # For each prime, update its multiples in the lpf array
    for p in primes:
        # Update all multiples of p: lpf[multiple] = p
        # This overwrites previous values, so the last (largest) prime wins
        multiples = range(p, MAX_VAL + 1, p)
        lpf[multiples] = p
    
    # Now find the n in [10000, 1000000] that minimizes max(lpf[n], lpf[n+1])
    # Vectorized approach:
    n_range = np.arange(10000, MAX_N + 1)
    lpf_n = lpf[n_range]
    lpf_np1 = lpf[n_range + 1]
    combined = np.maximum(lpf_n, lpf_np1)
    
    best_idx = np.argmin(combined)
    best_n = n_range[best_idx]
    
    return int(best_n)
# EVOLVE-BLOCK-END
