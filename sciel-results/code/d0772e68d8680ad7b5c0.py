import time
import numpy as np
from math import isqrt
from bisect import bisect_right

def _generate_powerful(limit):
    """Generate all powerful numbers up to limit.
    Powerful = a^2 * b^3 where b is square-free.
    """
    max_b = int(limit ** (1.0 / 3.0)) + 2
    is_sq_free = np.ones(max_b + 1, dtype=bool)
    is_sq_free[0] = False
    for i in range(2, isqrt(max_b) + 1):
        sq = i * i
        is_sq_free[sq::sq] = False
    is_sq_free[1] = True

    powerful = set()
    for b in range(1, max_b + 1):
        if not is_sq_free[b]:
            continue
        b3 = b ** 3
        if b3 > limit:
            break
        max_a = isqrt(limit // b3)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                powerful.add(val)
    return powerful

def _count_pairs_for_N(N, powerful_list, powerful_set, limit):
    """Count the number of pairs (a,b) with a<=b, a+b=N, both powerful."""
    a_min = max(1, N - limit)
    a_max = min(N // 2, limit)
    
    left = bisect_right(powerful_list, a_min - 1)
    right = bisect_right(powerful_list, a_max)
    
    count = 0
    for idx in range(left, right):
        a = powerful_list[idx]
        b = N - a
        if b in powerful_set:
            count += 1
    return count

def construct():
    t0 = time.time()
    deadline = t0 + 140  # Safety margin: 150s kill, leave 10s buffer
    
    best_result = None
    best_count = 0
    
    # Generate powerful numbers up to a large limit
    limit = 120_000_000
    powerful = _generate_powerful(limit)
    powerful_list = sorted(powerful)
    powerful_set = powerful
    n_powerful = len(powerful_list)
    
    if time.time() > deadline - 60:
        # Fallback: use FFT on smaller set
        limit_small = 5_000_000
        powerful_small = _generate_powerful(limit_small)
        powerful_list_small = sorted(powerful_small)
        powerful_set_small = powerful_small
        arr = np.zeros(limit_small + 1, dtype=np.float64)
        for p in powerful_small:
            arr[p] = 1.0
        fft_size = 1
        while fft_size < 2 * (limit_small + 1):
            fft_size *= 2
        F = np.fft.fft(arr, n=fft_size)
        conv = np.fft.ifft(F * F).real
        max_idx = int(np.argmax(conv[2:])) + 2
        N_best = max_idx
        pairs = []
        for a in powerful_list_small:
            if 2 * a > N_best:
                break
            b = N_best - a
            if b in powerful_set_small:
                pairs.append([a, b])
        return {'N': N_best, 'pairs': pairs}
    
    # Phase 1: Use FFT to find the top candidate N values
    # FFT size must be at least 2*limit+1
    fft_size = 1
    while fft_size < 2 * (limit + 1):
        fft_size *= 2
    
    arr = np.zeros(fft_size, dtype=np.float64)
    for p in powerful_list:
        arr[p] = 1.0
    
    F = np.fft.fft(arr)
    conv = np.fft.ifft(F * F).real
    conv = conv[:2 * limit + 1]
    
    # Find the top candidate N values by convolution value
    # Use a large number of top candidates
    top_k = 500
    indices = np.argsort(conv[2:])[-top_k:][::-1] + 2
    candidate_Ns = set(indices.tolist())
    
    # Also add some structured candidates
    # Multiples of highly composite numbers
    for m in range(1, 200):
        candidate_Ns.add(m * 100000)
    for m in range(1, 100):
        candidate_Ns.add(m * 50000)
    
    candidate_Ns = list(set(n for n in candidate_Ns if 2 <= n <= 2 * limit))
    
    # Sort by convolution value (descending) as a proxy for pair count
    candidate_Ns.sort(key=lambda n: -conv[n])
    
    # Phase 2: Test candidate N values, starting with most promising
    tested = set()
    for N in candidate_Ns:
        if time.time() > deadline - 20:
            break
        if N in tested:
            continue
        tested.add(N)
        
        count = _count_pairs_for_N(N, powerful_list, powerful_set, limit)
        
        if count > best_count:
            best_count = count
            pairs = []
            a_min = max(1, N - limit)
            a_max = min(N // 2, limit)
            left = bisect_right(powerful_list, a_min - 1)
            right = bisect_right(powerful_list, a_max)
            for idx in range(left, right):
                a = powerful_list[idx]
                b = N - a
                if b in powerful_set:
                    pairs.append([a, b])
            best_result = {'N': N, 'pairs': pairs}
    
    # Phase 3: If best is still low, do a broader sweep
    if best_result is None or best_count < 10:
        step = max(1, limit // 5000)
        for N in range(2 * step, 2 * limit, step):
            if time.time() > deadline - 15:
                break
            if N in tested:
                continue
            tested.add(N)
            
            count = _count_pairs_for_N(N, powerful_list, powerful_set, limit)
            
            if count > best_count:
                best_count = count
                pairs = []
                a_min = max(1, N - limit)
                a_max = min(N // 2, limit)
                left = bisect_right(powerful_list, a_min - 1)
                right = bisect_right(powerful_list, a_max)
                for idx in range(left, right):
                    a = powerful_list[idx]
                    b = N - a
                    if b in powerful_set:
                        pairs.append([a, b])
                best_result = {'N': N, 'pairs': pairs}
    
    if best_result is None:
        # Final fallback
        limit_small = 100_000
        powerful_small = _generate_powerful(limit_small)
        powerful_list_small = sorted(powerful_small)
        powerful_set_small = powerful_small
        arr = np.zeros(limit_small + 1, dtype=np.float64)
        for p in powerful_small:
            arr[p] = 1.0
        fft_size = 1
        while fft_size < 2 * (limit_small + 1):
            fft_size *= 2
        F = np.fft.fft(arr, n=fft_size)
        conv = np.fft.ifft(F * F).real
        max_idx = int(np.argmax(conv[2:])) + 2
        N_best = max_idx
        pairs = []
        for a in powerful_list_small:
            if 2 * a > N_best:
                break
            b = N_best - a
            if b in powerful_set_small:
                pairs.append([a, b])
        best_result = {'N': N_best, 'pairs': pairs}
    
    return best_result
