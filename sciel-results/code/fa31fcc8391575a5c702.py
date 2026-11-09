import numpy as np
import time

def generate_powerful(L):
    powerful = set()
    b_max = int(L ** (1/3))
    for b in range(1, b_max + 1):
        b3 = b ** 3
        a_max = int((L / b3) ** 0.5)
        for a in range(1, a_max + 1):
            p = a * a * b3
            powerful.add(p)
    return list(powerful)

def construct():
    start_time = time.time()
    deadline = 120.0 # 150s limit, keep margin
    
    # Choose L based on available time/memory
    # L=10^7 is safe
    L = 10**7
    
    P = generate_powerful(L)
    
    # Check time
    if time.time() - start_time > deadline - 10:
         # Fallback to smaller L
         L = 10**6
         P = generate_powerful(L)

    N_max = 2 * L
    size = 1
    while size < 2 * N_max:
        size *= 2
        
    sig = np.zeros(size, dtype=float)
    for p in P:
        sig[p] = 1.0
        
    fft_sig = np.fft.rfft(sig)
    conv = np.fft.irfft(fft_sig * np.conj(fft_sig))
    
    # Find N with max representations
    # We want to maximize the number of pairs a<=b
    # conv[N] counts ordered pairs.
    # Max of conv is a good proxy.
    max_idx = np.argmax(conv)
    N = int(max_idx)
    
    # Verify N is within range
    if N > 2 * L:
        N = 2 * L
        
    # Generate pairs
    P_set = set(P)
    P.sort()
    pairs = []
    for a in P:
        if a > N - a:
            break
        b = N - a
        if b in P_set:
            pairs.append([a, b])
            
    return {'N': N, 'pairs': pairs}
