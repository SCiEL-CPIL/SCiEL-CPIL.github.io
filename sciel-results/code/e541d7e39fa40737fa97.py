import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 60s limit
    
    N = 1000000
    LOW = 10000
    size = N + 2  # To cover n+1 up to 1000001
    
    # Initialize LPF array with zeros
    # lpf[i] will store the largest prime factor of i
    lpf = np.zeros(size, dtype=np.int32)
    
    # Sieve for Largest Prime Factor
    # We iterate p from 2 to size.
    # If lpf[p] is 0, p is prime (no smaller prime factor has marked it yet).
    # For each prime p, we mark all its multiples with p.
    # Since p increases, the last mark is from the largest prime factor.
    for p in range(2, size):
        if lpf[p] == 0:
            # Vectorized assignment: set lpf[j] = p for all j = p, 2p, 3p, ...
            lpf[p::p] = p
            
    if time.time() > deadline:
        return LOW
        
    # Vectorized scan for best consecutive pair
    # We want to minimize max(LPF(n), LPF(n+1)) for n in [LOW, N]
    # lpf[LOW:N] contains LPF(n) for n in [LOW, N-1]
    # lpf[LOW+1:N+1] contains LPF(n+1) for n in [LOW, N-1]
    # Note: The range of n is [10000, 1000000]. 
    # If n=1000000, n+1=1000001. So we need lpf up to 1000001.
    # The slice lpf[LOW:N] corresponds to n=LOW ... N-1.
    # The slice lpf[LOW+1:N+1] corresponds to n+1=LOW+1 ... N.
    # This covers n from LOW to N-1.
    # We need to include n=N.
    # Let's adjust slices to cover n in [LOW, N].
    # lpf[LOW:N+1] covers LPF(n) for n in [LOW, N].
    # lpf[LOW+1:N+2] covers LPF(n+1) for n in [LOW, N].
    
    scores = np.maximum(lpf[LOW:N+1], lpf[LOW+1:N+2])
    
    best_idx = int(np.argmin(scores))
    return LOW + best_idx
