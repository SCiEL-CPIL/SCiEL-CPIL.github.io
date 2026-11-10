"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    # Use a strict deadline to ensure we return within limits
    start_time = time.time()
    deadline = start_time + 55  # 55s out of 60s to be safe

    N = 123457

    # Sieve for Largest Prime Factor (LPF)
    # lpf[x] will store the largest prime factor of x
    # Initialize with 0
    lpf = [0] * (N + 1)
    
    # Sieve of Eratosthenes variant for LPF
    for i in range(2, N + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    # P(1) = 1
    lpf[1] = 1

    best_max_pf = float('inf')
    best_a = 1
    best_b = N - 1

    # We only need to check a <= b, so a in [1, N//2]
    # Since the objective max(P(a), P(b)) is symmetric, this is sufficient.
    for a in range(1, N // 2 + 1):
        if time.time() > deadline:
            break
            
        b = N - a
        # Get largest prime factors
        pa = lpf[a]
        pb = lpf[b]
        
        # Calculate max
        m = pa if pa > pb else pb
        
        if m < best_max_pf:
            best_max_pf = m
            best_a = a
            best_b = b

    return [best_a, best_b]
# EVOLVE-BLOCK-END
