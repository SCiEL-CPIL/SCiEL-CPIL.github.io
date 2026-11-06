import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin for 150s limit

    # Strategy:
    # 1. Generate powerful numbers up to L using a^2 * b^3 form.
    #    Every powerful number can be written as a^2 * b^3 for some a, b >= 1.
    #    (Proof: for n = prod p^{e_p}, e_p >= 2, write e_p = 2*alpha + 3*beta.)
    # 2. Use the pairwise-sum approach with numpy (inspiration 2) to count
    #    representations N = a + b with a <= b, both powerful.
    # 3. Find the N with maximum count.
    # 4. Extract and return pairs.
    #
    # The key structural choice: use the numpy matrix approach (P[:,None]+P[None,:])
    # rather than FFT convolution. This is simpler, avoids scipy dependency issues,
    # and for M ~ 10^4 powerful numbers, the M^2 matrix is ~800MB which is manageable.
    # Actually, let's be more careful about memory.
    #
    # For L = 10^8: number of powerful numbers is roughly L^(1/2) * (log L)^c...
    # Actually it's about sum_{b=1}^{L^(1/3)} sqrt(L/b^3) = sqrt(L) * sum_{b=1}^{L^(1/3)} b^{-3/2}
    # ≈ sqrt(L) * 2 = 2 * 10^4 for L = 10^8.
    # So M ≈ 20000, M^2 = 4*10^8, * 8 bytes = 3.2 GB. That's too much!
    #
    # Let's use L = 5*10^7 instead. Then sqrt(L) ≈ 7071, M ≈ 14000, M^2 ≈ 2*10^8, *8 = 1.6GB. Still a lot.
    #
    # Alternative: use the FFT approach from the current program but with a larger L.
    # Or use a chunked approach for the pairwise sums.
    #
    # Actually, let me reconsider. The current program uses FFT with L=2*10^7.
    # Let me use a hybrid: generate powerful numbers up to a larger L, then use
    # a memory-efficient counting method.
    #
    # Better idea: use the numpy approach but in chunks to avoid O(M^2) memory.
    # Or: since we only need the MAX count, we can use a different strategy.
    #
    # Actually, the simplest high-performance approach:
    # - Generate powerful numbers up to L (say L = 5*10^7)
    # - For counting, use np.add.outer in chunks, or use the FFT approach
    #
    # Let me use FFT (scipy) with a larger L than the current program.
    # Current uses L=2*10^7. Let me try L=5*10^7.
    # FFT of size 2*L = 10^8 requires ~10^8 * 8 bytes = 800MB for the array,
    # and scipy's fftconvolve will handle the rest.
    
    L = 5 * 10**7
    
    # Generate powerful numbers up to L
    powerfuls = set()
    max_b = int(L ** (1.0/3.0)) + 2
    for b in range(1, max_b + 1):
        b3 = b * b * b
        if b3 > L:
            break
        max_a = isqrt(L // b3)
        for a in range(1, max_a + 1):
            n = a * a * b3
            if n <= L:
                powerfuls.add(n)
    
    powerfuls_list = sorted(powerfuls)
    M = len(powerfuls_list)
    
    elapsed_gen = time.time() - start_time
    print(f"Generated {M} powerful numbers up to {L} in {elapsed_gen:.1f}s")
    
    if time.time() > deadline - 40:
        # Fallback: return a small known result
        return {'N': 2, 'pairs': [[1, 1]]}
    
    # Use FFT convolution to count pairs
    # arr[p] = 1 if p is powerful, else 0
    # convolve(arr, arr)[N] = number of ordered pairs (a,b) with a+b=N
    # For unordered with a<=b: if N is odd, count/2; if N even, (count+1)/2
    
    max_N = 2 * L
    arr = np.zeros(max_N + 1, dtype=np.float64)
    for p in powerfuls_list:
        arr[p] = 1.0
    
    elapsed_arr = time.time() - start_time
    print(f"Array setup done in {elapsed_arr:.1f}s")
    
    if time.time() > deadline - 30:
        # Try a smaller L with direct counting
        L_small = 10**7
        powerfuls_small = set()
        max_b_s = int(L_small ** (1.0/3.0)) + 2
        for b in range(1, max_b_s + 1):
            b3 = b * b * b
            if b3 > L_small:
                break
            max_a_s = isqrt(L_small // b3)
            for a in range(1, max_a_s + 1):
                n = a * a * b3
                if n <= L_small:
                    powerfuls_small.add(n)
        powerfuls_small = sorted(powerfuls_small)
        pows_set = set(powerfuls_small)
        best_N = 2
        best_count = 0
        for N in range(2, 2 * L_small + 1, 100):
            cnt = 0
            for a in powerfuls_small:
                if a > N // 2:
                    break
                if (N - a) in pows_set:
                    cnt += 1
            if cnt > best_count:
                best_count = cnt
                best_N = N
        pairs = []
        for a in powerfuls_small:
            if a > best_N // 2:
                break
            b_val = best_N - a
            if b_val in pows_set:
                pairs.append([a, b_val])
        return {'N': best_N, 'pairs': pairs}
    
    from scipy.signal import fftconvolve
    counts = fftconvolve(arr, arr, mode='full')
    
    elapsed_fft = time.time() - start_time
    print(f"FFT convolution done in {elapsed_fft:.1f}s")
    
    # Find best N
    powerfuls_set = set(powerfuls_list)
    best_N = 2
    best_count = 0
    for N in range(2, min(max_N + 1, len(counts))):
        ordered = int(round(counts[N]))
        if ordered == 0:
            continue
        if N % 2 == 0 and (N // 2) in powerfuls_set:
            unordered = (ordered + 1) // 2
        else:
            unordered = ordered // 2
        if unordered > best_count:
            best_count = unordered
            best_N = N
    
    elapsed_find = time.time() - start_time
    print(f"Best N = {best_N}, count = {best_count}, found in {elapsed_find:.1f}s")
    
    # Generate pairs for best_N
    pairs = []
    for a in powerfuls_list:
        if a > best_N // 2:
            break
        b = best_N - a
        if b in powerfuls_set:
            pairs.append([a, b])
    
    pairs.sort()
    
    elapsed_total = time.time() - start_time
    print(f"Total time: {elapsed_total:.1f}s")
    
    if not pairs:
        return {'N': 2, 'pairs': [[1, 1]]}
    
    return {'N': best_N, 'pairs': pairs}
