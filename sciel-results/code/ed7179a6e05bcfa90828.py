"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from bisect import bisect_left

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit using a^2 * b^3 form."""
    powerful = set()
    powerful.add(1)
    
    max_b = int(round(limit ** (1/3))) + 2
    
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = int(math.isqrt(limit // b3))
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    return sorted(powerful)

def construct():
    start_time = time.time()
    # Deadline guard: 150s limit, keep 15s safety margin
    deadline = start_time + 135.0
    
    # Phase 1: Generate powerful numbers up to a large limit
    # Use a large limit to maximize search space
    # Memory for FFT: 2*limit * 4 bytes (float32)
    # limit=80M -> 640MB for indicator array, plus FFT overhead
    # Let's try limit=100_000_000 first, fallback to 60M if memory/time issues
    
    limit = 100_000_000
    
    try:
        powerful = _generate_powerful(limit)
    except Exception:
        limit = 60_000_000
        powerful = _generate_powerful(limit)
    
    powerful_set = set(powerful)
    K = len(powerful)
    
    if K < 2:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Phase 2: Use FFT convolution to count all sums efficiently
    max_sum = 2 * limit
    n_fft = 1
    while n_fft < max_sum + 1:
        n_fft *= 2
    
    # Create indicator array
    f = np.zeros(n_fft, dtype=np.float32)
    for val in powerful:
        if val < n_fft:
            f[val] = 1.0
    
    # FFT convolution
    F = np.fft.rfft(f)
    conv = np.fft.irfft(F * F, n=n_fft)
    
    # Round to integers
    conv_int = np.round(conv[:max_sum + 1]).astype(np.int64)
    
    # Compute number of pairs with a <= b for each sum s
    # For even s where s/2 is powerful, add 1 before dividing by 2
    pairs_count = np.zeros(max_sum + 1, dtype=np.int64)
    
    even_s = np.arange(0, max_sum + 1, 2)
    half_s = even_s // 2
    is_powerful_half = (f[half_s] == 1.0)
    
    adjusted = conv_int.copy()
    adjusted[even_s[is_powerful_half]] += 1
    
    pairs_count = adjusted // 2
    
    # Exclude N=0 and N=1
    if max_sum >= 2:
        pairs_count[:2] = 0
    
    # Find top candidates using argpartition for speed
    top_k = 100
    if len(pairs_count) > top_k:
        top_indices = np.argpartition(pairs_count, -top_k)[-top_k:]
    else:
        top_indices = np.arange(len(pairs_count))
    
    # Sort these top indices by their count value in descending order
    top_indices = top_indices[np.argsort(pairs_count[top_indices])[::-1]]
    
    # Verify top candidates and find the best
    best_N = 2
    best_count = 1
    best_pairs = [[1, 1]]
    
    for idx in top_indices:
        if time.time() > deadline - 10:
            break
        
        N = int(idx)
        if N < 2:
            continue
        
        # Count pairs for this N
        count = 0
        exact_pairs = []
        for a in powerful:
            if a > N // 2:
                break
            b = N - a
            if b in powerful_set:
                count += 1
                exact_pairs.append([a, b])
        
        if count > best_count:
            best_count = count
            best_N = N
            best_pairs = exact_pairs
        
        # If the conv value for this candidate is lower than our current best_count * 2 - 1,
        # and we've processed a few, we might be able to stop?
        # conv[idx] is an upper bound on 2*count (roughly).
        # Since top_indices is sorted by pairs_count descending, if we hit this condition, we can break.
        if pairs_count[idx] < best_count:
            break
    
    return {'N': best_N, 'pairs': best_pairs}
# EVOLVE-BLOCK-END
