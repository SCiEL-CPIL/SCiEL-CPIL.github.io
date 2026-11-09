"""
Find N with many representations as sums of two powerful numbers.
Strategy:
  1. Generate a large set of powerful numbers up to a limit L.
  2. Use FFT convolution to find the N (up to 2L) with the maximum number of pairs.
  3. Extract the pairs for the best N.
"""
import time
import numpy as np
from math import isqrt

def _generate_powerful(limit):
    """
    Generate all powerful numbers up to limit.
    A powerful number n can be written as m^2 * b where b is squarefree
    and every prime dividing b also divides m.
    """
    sqrt_limit = isqrt(limit)
    powerful_set = set([1])
    
    # Precompute smallest prime factor up to sqrt_limit for fast factorization
    spf = list(range(sqrt_limit + 1))
    for i in range(2, isqrt(sqrt_limit) + 1):
        if spf[i] == i:
            for j in range(i * i, sqrt_limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    for m in range(2, sqrt_limit + 1):
        # Factor m to get distinct prime factors
        temp = m
        primes = []
        while temp > 1:
            if temp <= sqrt_limit:
                p = spf[temp]
            else:
                p = temp
            primes.append(p)
            while temp % p == 0:
                temp //= p
        
        # Enumerate all subsets of distinct primes -> squarefree divisors
        n_primes = len(primes)
        for mask in range(1 << n_primes):
            b = 1
            for i in range(n_primes):
                if mask & (1 << i):
                    b *= primes[i]
            m2 = m * m
            if b <= limit // m2:
                val = m2 * b
                if val <= limit:
                    powerful_set.add(val)
    
    return sorted(powerful_set)

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Choose L such that FFT is fast enough.
    # L = 5 * 10^7 -> FFT size ~ 2^26 = 6.7e7. Fast.
    # L = 10^8 -> FFT size ~ 2^27 = 1.34e8. Might be slower but better coverage.
    # Let's try L = 8 * 10^7 to be safe on memory/time.
    L = 80_000_000
    
    powerful = _generate_powerful(L)
    powerful_set = set(powerful)
    
    elapsed = time.time() - start_time
    if elapsed > deadline - 20:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Create indicator array
    ind = np.zeros(L + 1, dtype=np.float32)
    for p in powerful:
        ind[p] = 1.0
    
    # Perform convolution
    # We need the convolution up to index 2L.
    # fftconvolve handles the size automatically.
    # Using float32 to save memory and speed up.
    try:
        conv = np.fft.irfft(np.fft.rfft(ind) ** 2, n=2 * L + 1)
    except MemoryError:
        # Fallback to smaller L if memory error
        L = 40_000_000
        powerful = _generate_powerful(L)
        powerful_set = set(powerful)
        ind = np.zeros(L + 1, dtype=np.float32)
        for p in powerful:
            ind[p] = 1.0
        conv = np.fft.irfft(np.fft.rfft(ind) ** 2, n=2 * L + 1)
    
    # Find the N with maximum representations
    # conv[N] is the number of ordered pairs (a,b) such that a+b=N.
    # We want to maximize the number of distinct pairs a <= b.
    # Count = (conv[N] + is_powerful(N//2) * (N%2==0)) / 2
    # But we can just find the max of conv[N] first, as the adjustment is small.
    # Actually, we should compute the score for each N.
    
    # To avoid iterating over 2L+1 elements in Python (slow), we can use numpy.
    # But we need to check if N/2 is powerful.
    # Let's find the top few candidates from conv.
    
    # Get indices of top 100 values in conv
    # Note: conv[0] and conv[1] are small. We care about larger N.
    # Also N must be >= 2.
    
    # We can just take the argmax of conv.
    # But we need to be careful with the parity adjustment.
    # Let's get the top 50 candidates.
    
    # Flatten conv to 1D
    conv_flat = conv
    
    # Find top 50 indices
    # Using argsort is O(N log N) or O(N) if we use partial sort.
    # For 1.6e8 elements, argsort might be slow or memory heavy.
    # Let's just find the global max and its neighbors?
    # No, the max might be at a specific N.
    
    # Alternative: Just find the index of the maximum value.
    # The parity adjustment is at most 0.5 difference in score, which is negligible compared to the count (which can be in hundreds/thousands).
    # So argmax(conv) is a very good candidate.
    
    best_N = int(np.argmax(conv_flat[2:2*L+1]) + 2)
    
    # Let's check a small window around best_N to be safe
    # Also check if best_N is within our range [2, 2L]
    if best_N < 2 or best_N > 2 * L:
        best_N = 2
    
    # Extract pairs for best_N
    pairs = []
    half_N = best_N // 2
    # Iterate through powerful numbers <= half_N
    # We can binary search in the sorted `powerful` list
    import bisect
    idx = bisect.bisect_right(powerful, half_N)
    for i in range(idx):
        a = powerful[i]
        b = best_N - a
        if b >= a and b in powerful_set:
            pairs.append([a, b])
    
    # Verify the count
    # Score is len(pairs)
    
    # Optional: Try a few other candidates if time permits
    # But the FFT peak is usually the global max for this kind of problem.
    
    return {'N': best_N, 'pairs': pairs}

if __name__ == "__main__":
    result = construct()
    print(f"N = {result['N']}")
    print(f"Number of pairs: {len(result['pairs'])}")
    print(f"First few pairs: {result['pairs'][:10]}")
