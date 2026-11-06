# EVOLVE-BLOCK-START
import time
import numpy as np
from math import isqrt

def _gen_powerful(limit):
    """Generate all powerful numbers up to limit using square-free sieve."""
    if limit < 1:
        return set()
    # Sieve for square-free numbers
    is_sqfree = [True] * (limit + 1)
    is_sqfree[0] = False
    # Sieve of Eratosthenes for primes up to sqrt(limit)
    sqrt_lim = isqrt(limit) + 1
    if sqrt_lim >= 2:
        sieve = [True] * (sqrt_lim + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, isqrt(sqrt_lim) + 1):
            if sieve[i]:
                for j in range(i * i, sqrt_lim + 1, i):
                    sieve[j] = False
        primes = [i for i in range(2, sqrt_lim + 1) if sieve[i]]
    else:
        primes = []
    
    # Mark non-square-free numbers
    for p in primes:
        p2 = p * p
        if p2 > limit:
            break
        for j in range(p2, limit + 1, p2):
            is_sqfree[j] = False
    
    # Generate powerful numbers: n = a^2 * b where b is square-free
    powerful = set()
    powerful.add(1)
    for b in range(1, limit + 1):
        if is_sqfree[b]:
            max_a = isqrt(limit // b)
            for a in range(1, max_a + 1):
                val = a * a * b
                powerful.add(val)
    return powerful

def construct():
    start_time = time.time()
    deadline = 110  # safety margin
    
    best_N = 2
    best_pairs = [[1, 1]]
    best_count = 1
    
    # Try increasing limits
    limits = [200_000, 500_000, 1_000_000, 2_000_000]
    
    for L in limits:
        if time.time() - start_time > deadline - 20:
            break
        try:
            pow_set = _gen_powerful(L)
            pow_list = sorted(pow_set)
            
            # FFT convolution
            n_fft = 1
            while n_fft < 2 * L + 1:
                n_fft *= 2
            
            arr = np.zeros(n_fft, dtype=np.float64)
            for x in pow_list:
                arr[x] = 1.0
            
            fft_arr = np.fft.fft(arr)
            conv = np.fft.ifft(fft_arr * fft_arr).real
            
            # Find best N
            for N in range(2, 2 * L + 1):
                c = conv[N]
                if N % 2 == 0:
                    half = N // 2
                    self_c = 1 if half in pow_set else 0
                else:
                    self_c = 0
                unordered = (c + self_c) / 2.0
                if unordered > best_count:
                    best_count = unordered
                    best_N = N
            
            # Extract pairs for best_N
            pairs = []
            for a in pow_list:
                if a > best_N:
                    break
                b = best_N - a
                if b >= a and b in pow_set:
                    pairs.append([a, b])
            best_pairs = pairs
            
        except Exception:
            break
    
    return {'N': best_N, 'pairs': best_pairs}
# EVOLVE-BLOCK-END
