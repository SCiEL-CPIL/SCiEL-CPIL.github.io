"""Optimized program to find long runs of consecutive integers with distinct totients."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np

def sieve_primes(limit):
    """Generate all primes up to limit using Sieve of Eratosthenes."""
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(limit + 1) if is_prime[i]]

def segmented_spf(start, length, primes):
    """Build smallest prime factor array for [start, start+length-1] using segmented sieve."""
    spf = np.zeros(length, dtype=np.int64)
    # Initialize spf to 0 (meaning "not yet assigned")
    for p in primes:
        if p * p > start + length - 1:
            break
        # Find the first multiple of p in [start, start+length-1]
        first = ((start + p - 1) // p) * p
        for j in range(first, start + length, p):
            idx = j - start
            if spf[idx] == 0:
                spf[idx] = p
    return spf

def phi_from_spf(n, spf_idx, start, length, primes):
    """Compute phi(n) where n = start + spf_idx, using the segmented SPF array."""
    result = n
    temp = n
    # Follow SPF chain for factors found in the segmented sieve
    while temp > 1 and spf_idx < length:
        p = 0
        # Check if temp has a small prime factor from our SPF
        # Actually, we need to factor temp properly
        break
    # This approach is tricky; let's use a different method
    return result

def phi_window(s, k, primes):
    """Compute phi(s), phi(s+1), ..., phi(s+k-1) using segmented SPF."""
    # Build segmented SPF for the window
    spf = np.zeros(k, dtype=np.int64)
    max_val = s + k - 1
    for p in primes:
        if p * p > max_val:
            break
        first = ((s + p - 1) // p) * p
        for j in range(first, s + k, p):
            idx = j - s
            if spf[idx] == 0:
                spf[idx] = p
    
    # Now compute phi for each number
    phis = np.zeros(k, dtype=np.int64)
    for i in range(k):
        n = s + i
        result = n
        temp = n
        # Factor temp using SPF for small factors, then trial division for remaining
        if spf[i] != 0:
            p = spf[i]
            result -= result // p
            while temp % p == 0:
                temp //= p
        # For remaining temp, try small primes
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                result -= result // p
                while temp % p == 0:
                    temp //= p
        if temp > 1:
            result -= result // temp
        phis[i] = result
    return phis

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s limit
    
    # Generate primes up to 10^6
    prime_limit = 10**6
    primes = sieve_primes(prime_limit)
    primes_arr = np.array(primes, dtype=np.int64)
    
    def check_run(s, k, primes):
        """Check if phi(s), ..., phi(s+k-1) are all distinct. Returns True if valid."""
        phis = phi_window(s, k, primes)
        # Check for duplicates
        if len(set(phis.tolist())) == k:
            return True
        return False
    
    best_result = {'start': 1, 'length': 1}
    
    # Try k=200 at many candidate positions
    # For large s, k=200 should almost always work
    candidates = []
    
    # Near the top boundary
    base = 10**12 - 200
    for offset in range(0, 10000, 10):
        s = base - offset
        if s >= 1:
            candidates.append(s)
    
    # Spread across the range
    for frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        s = int(10**12 * frac)
        candidates.append(s)
        for delta in range(-5000, 5001, 50):
            s2 = s + delta
            if 1 <= s2 <= 10**12 - 200:
                candidates.append(s2)
    
    # Pseudo-random candidates
    import random
    rng = random.Random(42)
    for _ in range(200):
        s = rng.randint(1, 10**12 - 200)
        candidates.append(s)
    
    candidates = list(set(candidates))
    candidates.sort(reverse=True)
    
    # Try k=200 first
    for s in candidates:
        if time.time() > deadline - 20:
            break
        if s + 199 > 10**12:
            continue
        valid = check_run(s, 200, primes)
        if valid:
            return {'start': s, 'length': 200}
    
    # If k=200 wasn't found, try smaller k
    for k in [199, 198, 195, 190, 180, 170, 150, 100, 50, 25, 10, 5, 3, 2, 1]:
        if time.time() > deadline - 5:
            break
        for s in candidates:
            if s + k - 1 > 10**12 or s < 1:
                continue
            valid = check_run(s, k, primes)
            if valid:
                return {'start': s, 'length': k}
    
    return best_result

# EVOLVE-BLOCK-END
