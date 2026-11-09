"""Find N ≤ 10^18 with many representations as sum of two powerful numbers."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from math import isqrt

def _is_squarefree(n):
    """Check if n is squarefree using trial division up to sqrt(n)."""
    if n <= 1:
        return True
    temp = n
    p = 2
    while p * p <= temp:
        if temp % (p * p) == 0:
            return False
        p += 1 if p == 2 else 2
    return True

def _generate_powerful(limit):
    """
    Generate all powerful numbers up to limit.
    A powerful number is of the form a^2 * b^3 where b is squarefree.
    """
    result = set()
    # b must be squarefree and b^3 <= limit
    max_b = int(limit ** (1/3)) + 2
    for b in range(1, max_b + 1):
        if not _is_squarefree(b):
            continue
        b3 = b ** 3
        if b3 > limit:
            continue
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                result.add(val)
    return sorted(result)

def construct():
    start_time = time.time()
    # Safety margin: 150s limit, keep 10s buffer
    deadline = start_time + 140.0
    
    # Phase 1: Generate powerful numbers
    # Based on findings, 5*10^7 gives ~22k numbers, which is manageable for FFT
    # and allows N up to 10^8.
    # We can try to go higher if time permits, but 5*10^7 is a solid baseline.
    pw_limit = 5 * 10**7
    
    powerful = _generate_powerful(pw_limit)
    n_pw = len(powerful)
    
    if time.time() > deadline - 30:
        # Fallback if generation took too long (unlikely for 5*10^7)
        return {'N': 2, 'pairs': [[1, 1]]}

    # Phase 2: FFT Convolution
    # We want to count pairs (a, b) with a + b = N.
    # Let P be an array where P[x] = 1 if x is in powerful, else 0.
    # The convolution C = P * P gives C[N] = sum P[a]P[N-a] = count of pairs.
    # Size of array needs to be at least 2 * max(powerful) to avoid circular convolution artifacts.
    # We use next power of 2 for FFT efficiency.
    
    max_val = max(powerful) if powerful else 1
    # N can be up to 2 * max_val
    size = 1
    while size < 2 * max_val + 1:
        size *= 2
    
    # Create the signal
    signal = np.zeros(size, dtype=np.float64)
    for p in powerful:
        signal[p] = 1.0
        
    # Perform FFT
    # Use rfft for real input, which is faster
    fft_signal = np.fft.rfft(signal)
    # Convolution in frequency domain
    conv_fft = fft_signal * fft_signal
    # Inverse FFT
    conv_result = np.fft.irfft(conv_fft, n=size)
    
    # conv_result[N] is the count of ordered pairs (a, b) such that a+b=N.
    # We need a <= b.
    # If a != b, (a,b) and (b,a) are both counted. So count_unordered = count_ordered / 2.
    # If a == b (i.e., N is even and N/2 is powerful), it is counted once.
    # So, number of pairs with a <= b is:
    #   if N is even and N/2 in powerful: (conv[N] + 1) / 2
    #   else: conv[N] / 2
    # Actually, simpler:
    # Let c = conv[N].
    # Let self_count = 1 if (N % 2 == 0 and N//2 in powerful_set) else 0.
    # Pairs a <= b: (c - self_count) / 2 + self_count = (c + self_count) / 2.
    
    # Find the N with maximum representations.
    # We are interested in N >= 2.
    # We can just find the max in the valid range.
    # Note: conv_result might have small floating point errors, but counts are integers.
    
    # To be safe and fast, we can just take the argmax of the real part.
    # We restrict to N > 0.
    valid_range = conv_result[2:]
    if len(valid_range) == 0:
         return {'N': 2, 'pairs': [[1, 1]]}
         
    best_idx = np.argmax(valid_range)
    best_N = best_idx + 2
    
    # Check time
    if time.time() > deadline - 10:
        # Just return the count as a proxy? No, we need pairs.
        # If we don't have time to extract pairs, we might fail validation if we return wrong pairs.
        # But extraction is O(K), which is fast.
        pass

    # Phase 3: Extract pairs for best_N
    powerful_set = set(powerful)
    pairs = []
    for a in powerful:
        if a > best_N - a:
            break
        b = best_N - a
        if b in powerful_set:
            pairs.append([a, b])
            
    # Verify
    if not pairs:
        # Should not happen if best_N was derived from the convolution correctly
        # Fallback to a simple case
        return {'N': 2, 'pairs': [[1, 1]]}

    return {'N': best_N, 'pairs': pairs}

# EVOLVE-BLOCK-END
