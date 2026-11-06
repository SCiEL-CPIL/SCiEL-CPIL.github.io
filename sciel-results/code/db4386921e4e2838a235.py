"""Find N <= 10^18 with many representations as sum of two powerful numbers."""
import time
import random
from math import isqrt
import numpy as np

def _is_square_free(n):
    """Check if n is square-free."""
    if n < 2:
        return True
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in small_primes:
        if p * p > n:
            break
        if n % (p * p) == 0:
            return False
        while n % p == 0:
            n //= p
    if n == 1:
        return True
    r = isqrt(n)
    if r * r == n:
        return False
    return True

def _generate_powerful(limit):
    """Generate all powerful numbers <= limit.
    Powerful numbers are of the form a^2 * b^3 where b is square-free.
    """
    powerful = set([1])
    
    max_b = int(limit ** (1/3)) + 2
    
    # Sieve for square-free numbers up to max_b
    sq_free = bytearray(b'\x01') * (max_b + 1)
    sq_free[0] = 0
    if max_b >= 1:
        sq_free[1] = 1
    
    for p in range(2, int(limit ** (1/6)) + 2):
        if sq_free[p]:
            p2 = p * p
            for k in range(p2, max_b + 1, p2):
                sq_free[k] = 0
    
    for b in range(1, max_b + 1):
        if not sq_free[b]:
            continue
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
            else:
                break
    
    return sorted(powerful)

def _count_reps(N, powerful_list, powerful_set):
    """Count the number of pairs (a, b) with a <= b, a + b = N, a, b in powerful_set."""
    count = 0
    pairs = []
    for a in powerful_list:
        if 2 * a > N:
            break
        b = N - a
        if b in powerful_set:
            count += 1
            pairs.append([a, b])
    return count, pairs

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s buffer before 150s limit
    
    # Strategy: Use FFT to find N with maximum representations
    # Use float32 to allow larger range
    limit_pw = 10**8  # ~2-3 million powerful numbers
    
    powerful = _generate_powerful(limit_pw)
    powerful_set = set(powerful)
    n_pw = len(powerful)
    
    elapsed = time.time() - start_time
    
    # Use FFT convolution to count representations for all N simultaneously
    max_N = 2 * limit_pw
    # Use next power of 2 for FFT efficiency
    fft_size = 1
    while fft_size < max_N + 1:
        fft_size *= 2
    
    # Create signal with float32 for larger arrays
    signal = np.zeros(fft_size, dtype=np.float32)
    for pw in powerful:
        signal[pw] = 1.0
    
    # FFT convolution using float32
    fft_signal = np.fft.rfft(signal)
    fft_conv = fft_signal * fft_signal
    conv_result = np.fft.irfft(fft_conv, n=fft_size)
    
    # Find the N with maximum representations
    search_max = min(max_N, 10**18)
    
    # Get top candidates from convolution - examine more candidates
    # conv_result[i] gives count of ordered pairs summing to i
    top_indices = np.argsort(conv_result[2:search_max+1])[::-1][:200] + 2
    
    best_n = None
    best_count = 0
    best_pairs = []
    
    for idx in top_indices:
        if time.time() > deadline - 10:
            break
        n_candidate = int(idx)
        count, pairs = _count_reps(n_candidate, powerful, powerful_set)
        if count > best_count:
            best_count = count
            best_n = n_candidate
            best_pairs = pairs
    
    if best_n is not None and best_pairs:
        return {'N': best_n, 'pairs': best_pairs}
    
    # Fallback: random search
    random.seed(42)
    for _ in range(100000):
        if time.time() > deadline - 5:
            break
        idx1 = random.randint(0, n_pw - 1)
        idx2 = random.randint(0, n_pw - 1)
        n_candidate = powerful[idx1] + powerful[idx2]
        
        if n_candidate > 10**18 or n_candidate < 2:
            continue
        
        count, pairs = _count_reps(n_candidate, powerful, powerful_set)
        
        if count > best_count:
            best_count = count
            best_n = n_candidate
            best_pairs = pairs
    
    if best_n is not None and best_pairs:
        return {'N': best_n, 'pairs': best_pairs}
    
    return {'N': 2, 'pairs': [[1, 1]]}
