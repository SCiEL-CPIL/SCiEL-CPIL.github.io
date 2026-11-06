import numpy as np
import time
import math

def construct():
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    # 1. Determine M and FFT size
    # We want to maximize M within memory/time limits.
    # Let's try M = 20,000,000 (2 * 10^7)
    M = 20_000_000
    
    # Check time
    if time.time() - start_time > deadline:
        return {'N': 2, 'pairs': [[1, 1]]}
        
    # 2. Generate powerful numbers up to M
    powerful_set = set()
    powerful_set.add(1)
    
    max_b = int(round(M ** (1/3)))
    for b in range(1, max_b + 1):
        b3 = b ** 3
        if b3 > M:
            break
        max_a = int(math.isqrt(M // b3))
        for a in range(1, max_a + 1):
            val = (a * a) * b3
            if val <= M:
                powerful_set.add(val)
                
    powerful_list = sorted(powerful_set)
    
    # 3. Prepare for FFT
    # We need convolution of indicator vector with itself.
    # Size must be at least 2*M.
    # Next power of 2.
    size = 1
    while size < 2 * M + 1:
        size *= 2
        
    # Check memory feasibility roughly
    # size * 16 bytes
    # For M=20M, size=2^26=67M, mem ~ 1GB. OK.
    
    arr = np.zeros(size, dtype=np.float64)
    for p in powerful_list:
        arr[p] = 1.0
        
    # 4. FFT
    fft_arr = np.fft.fft(arr)
    conv_fft = fft_arr * fft_arr
    conv = np.fft.ifft(conv_fft).real
    
    # 5. Find max
    # The valid range for N is up to 2*M.
    # Also N >= 2 (since 1+1=2).
    # Find index of max value in conv[2:2*M+1]
    
    # To avoid index issues, just find global max and check bounds
    max_val = np.max(conv)
    N = int(np.argmax(conv))
    
    # Ensure N is valid
    if N < 2 or N > 2 * M:
         # Fallback to a known small one if something went wrong
         N = 2
         max_val = 1
         
    # 6. Extract pairs
    # We need to list pairs (a, b) with a <= b, a+b=N, a,b powerful.
    pairs = []
    powerful_set_check = powerful_set # Use set for O(1) lookup
    
    # Iterate through powerful numbers <= N/2
    for a in powerful_list:
        if a > N // 2:
            break
        b = N - a
        if b in powerful_set_check:
            pairs.append([a, b])
            
    # Verify count matches max_val (approximately, due to floating point errors in FFT)
    # FFT results are floating point, so max_val might be 11.999999 or 12.000001
    # The actual count is len(pairs)
    
    return {'N': N, 'pairs': pairs}
