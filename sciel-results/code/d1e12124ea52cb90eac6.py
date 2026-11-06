import math
import time
import bisect

def sieve_primes(limit):
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

PRIMES = sieve_primes(10**7)

def is_p_smooth_fast(n, p, primes, p_idx):
    """
    Check if n is p-smooth.
    p_idx is the index in primes such that primes[p_idx-1] <= p < primes[p_idx]
    We iterate primes[0] to primes[p_idx-1].
    """
    n_rem = n
    # Iterate only primes <= p
    for i in range(p_idx):
        q = primes[i]
        if q * q > n_rem:
            break
        while n_rem % q == 0:
            n_rem //= q
    return n_rem == 1

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_k = 0
    best_p = 2
    
    # Precompute indices for primes?
    # We iterate p in PRIMES.
    # For each p, p_idx is just the current index i in the loop if we iterate over enumerate.
    
    # Let's iterate over indices to get p and its index in PRIMES
    # But PRIMES contains all primes up to 10^7.
    # We want to check p up to some limit fully.
    
    # Phase 1: Check all primes up to 2,000,000
    # Find index of 2,000,000 in PRIMES
    limit_p = 2 * 10**6
    max_idx = bisect.bisect_right(PRIMES, limit_p)
    
    for i in range(max_idx):
        if time.time() > deadline:
            break
        p = PRIMES[i]
        p2 = p * p
        k = 0
        
        for j in range(1, 201):
            if time.time() > deadline:
                break
            n_val = p2 + j
            if is_p_smooth_fast(n_val, p, PRIMES, i+1):
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
            
    # Phase 2: If time remains, check larger primes with limited k
    # For p > 2M, cost is high.
    # Let's say we check up to k=20 for these.
    if time.time() < deadline - 5:
        # Iterate from max_idx to end of PRIMES
        # But step? No, just iterate.
        # Stop if time runs out.
        for i in range(max_idx, len(PRIMES)):
            if time.time() > deadline:
                break
            p = PRIMES[i]
            p2 = p * p
            k = 0
            # Check up to k=20 or until fail
            for j in range(1, 21):
                if time.time() > deadline:
                    break
                n_val = p2 + j
                if is_p_smooth_fast(n_val, p, PRIMES, i+1):
                    k += 1
                else:
                    break
            
            if k > best_k:
                best_k = k
                best_p = p

    return {'p': best_p, 'k': best_k}
