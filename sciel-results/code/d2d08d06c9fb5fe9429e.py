import time
import numpy as np
from scipy.signal import fftconvolve
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 105  # 15s safety margin for 120s limit
    
    # Generate powerful numbers up to L using vectorized NumPy
    # Powerful numbers: a^2 * b^3 where a^2*b^3 <= L
    L = 3 * 10**8
    max_b = int(L ** (1.0/3.0)) + 1  # ~670
    
    powerfuls_list = []
    for b in range(1, max_b + 1):
        b3 = b ** 3
        max_a = isqrt(L // b3)
        if max_a < 1:
            break
        # Vectorized: a^2 * b3 for a = 1..max_a
        a_vals = np.arange(1, max_a + 1, dtype=np.int64)
        vals = a_vals * a_vals * b3
        vals = vals[vals <= L]
        if len(vals) > 0:
            powerfuls_list.append(vals)
    
    if not powerfuls_list:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    powerfuls_arr = np.concatenate(powerfuls_list)
    # Remove duplicates
    powerfuls_arr = np.unique(powerfuls_arr)
    powerfuls_arr = powerfuls_arr[powerfuls_arr <= L]
    
    elapsed_gen = time.time() - start_time
    
    # Adaptive L: if generation took too long, reduce effective L for FFT
    # FFT of size N_fft takes roughly O(N_fft log N_fft)
    # For N_fft = 2*(L+1), we estimate time
    remaining = deadline - time.time()
    # Empirically, FFT of size 2*10^8 takes ~5-8s on typical hardware
    # Be conservative: if remaining < 30s, reduce L
    if remaining < 30:
        # Scale L down proportionally
        scale = max(0.3, remaining / 50.0)
        L_eff = int(L * scale)
        powerfuls_arr = powerfuls_arr[powerfuls_arr <= L_eff]
    else:
        L_eff = int(L)
    
    if len(powerfuls_arr) == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Check if we have enough time for FFT
    if time.time() > deadline - 20:
        # Fallback: sparse search without FFT
        powerfuls_set = set(powerfuls_arr.tolist())
        powerfuls_sorted = powerfuls_arr.tolist()
        best_N = 2
        best_count = 0
        # Sample candidate N values from sums of nearby powerful numbers
        candidates = set()
        n_samples = min(2000, len(powerfuls_sorted))
        step = max(1, len(powerfuls_sorted) // n_samples)
        for i in range(0, len(powerfuls_sorted), step):
            candidates.add(powerfuls_sorted[i] * 2)
            if i + 1 < len(powerfuls_sorted):
                candidates.add(powerfuls_sorted[i] + powerfuls_sorted[i+1])
        for N in candidates:
            if N < 2 or N > 2 * L_eff:
                continue
            count = 0
            for a in powerfuls_sorted:
                if a > N // 2:
                    break
                if (N - a) in powerfuls_set:
                    count += 1
            if count > best_count:
                best_count = count
                best_N = N
        pairs = []
        for a in powerfuls_sorted:
            if a > best_N // 2:
                break
            b = best_N - a
            if b in powerfuls_set:
                pairs.append([a, b])
        pairs.sort()
        return {'N': best_N, 'pairs': pairs}
    
    # Create dense array for FFT
    size = L_eff + 1
    arr = np.zeros(size, dtype=np.int64)
    arr[powerfuls_arr[powerfuls_arr < size]] = 1
    
    # Compute convolution to count representations
    counts = fftconvolve(arr, arr, mode='full')
    
    # Find best N in range [2, 2*L_eff]
    search_slice = counts[2:2*L_eff+1]
    if len(search_slice) == 0:
        return {'N': 2, 'pairs': [[1, 1]]}
    best_N = int(np.argmax(search_slice) + 2)
    
    # Generate pairs for best_N
    powerfuls_set = set(powerfuls_arr.tolist())
    powerfuls_sorted = powerfuls_arr.tolist()
    
    pairs = []
    max_a = best_N // 2
    
    for a in powerfuls_sorted:
        if a > max_a:
            break
        b = best_N - a
        if b in powerfuls_set:
            pairs.append([a, b])
            
    pairs.sort()
    
    return {'N': best_N, 'pairs': pairs}
# EVOLVE-BLOCK-END
