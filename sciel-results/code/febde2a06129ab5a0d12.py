import time
import numpy as np
from scipy.signal import fftconvolve

def is_squarefree(n):
    if n <= 1:
        return True
    p = 2
    while p * p <= n:
        if n % (p * p) == 0:
            return False
        p += 1
    return True

def generate_powerful(limit):
    pows = set()
    max_b = int(limit ** (1/3)) + 1
    for b in range(1, max_b):
        if not is_squarefree(b):
            continue
        b3 = b ** 3
        max_a = int((limit // b3) ** 0.5)
        for a in range(1, max_a + 1):
            val = a * a * b3
            if val <= limit:
                pows.add(val)
    return pows

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    # Try to find the best N up to L
    # L=10^8 is feasible with FFT
    L = 100000000
    
    # Generate powerful numbers up to L
    # Note: We need powerful numbers up to N, where N can be up to 2L
    # But a+b=N implies a,b <= N.
    # If we search N up to 2L, we need powerful numbers up to 2L.
    # However, if a > L, then b = N-a < L.
    # So we just need powerful numbers up to 2L?
    # Actually, if N <= 2L, and a,b are powerful, max(a,b) can be close to 2L.
    # So generate up to 2L.
    
    gen_limit = 2 * L
    pows = generate_powerful(gen_limit)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        # Fallback to smaller search if generation was too slow (unlikely)
        L = 1000000
        gen_limit = 2 * L
        pows = generate_powerful(gen_limit)

    # Create array for FFT
    # Size needs to be at least 2 * max_val + 1
    # max_val is 2L
    # Convolution result size is 2 * (2L) + 1 = 4L + 1
    # We only care about sums up to 2L (since a,b <= 2L? No, a,b <= gen_limit=2L)
    # Wait, if a,b <= 2L, sum can be up to 4L.
    # But we want N <= 10^18.
    # If we restrict search to N <= 2L, we are safe.
    # Let's search N in [0, 2L].
    
    # Array A of size 2L+1
    # A[x] = 1 if x is powerful, 0 otherwise
    size = 2 * L + 1
    A = np.zeros(size, dtype=np.int64)
    for p in pows:
        if p < size:
            A[p] = 1
    
    # Perform convolution
    # C = A * A
    # C[k] is the number of pairs (a,b) such that a+b=k
    C = fftconvolve(A, A)
    
    # We want to maximize the number of distinct pairs a <= b.
    # C[N] counts ordered pairs.
    # Distinct pairs count is roughly C[N] / 2.
    # We can just find N with max C[N].
    
    # Restrict search to N <= 2L?
    # The convolution C has length 2*size - 1 = 4L + 1.
    # We should probably look for N up to 2L.
    # Actually, if we generated pows up to 2L, we can form sums up to 4L.
    # But larger N might have more representations?
    # Let's look at the peak of C.
    
    # Find the index with maximum value
    # We can ignore very small N.
    # Let's find the global max in C.
    max_idx = np.argmax(C)
    N = int(max_idx)
    
    # Verify N is within bounds and valid
    # The task says N <= 10^18. 2L = 2*10^8, so we are safe.
    
    # Extract pairs
    pows_set = pows # It's a set
    pairs = []
    # Iterate through powerful numbers a
    # We need a <= b => 2a <= N
    for a in pows:
        if 2 * a > N:
            break
        b = N - a
        if b in pows_set:
            pairs.append([a, b])
    
    # Sort pairs for consistency (optional but good)
    pairs.sort()
    
    return {'N': N, 'pairs': pairs}
