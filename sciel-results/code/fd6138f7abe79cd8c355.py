"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Find two positive integers a, b with a + b = 123457 that minimize max(P(a), P(b)),
    where P(n) is the largest prime factor of n, and P(1) = 1.
    
    Strategy:
    1. Precompute the largest prime factor (LPF) for all integers up to 123457 using a sieve.
    2. Perform a linear scan over all splits a from 1 to 123456, computing b = 123457 - a.
    3. Track the split that minimizes max(LPF[a], LPF[b]).
    
    This guarantees the global optimum in O(N) time where N = 123457.
    """
    N = 123457
    
    # Deadline guard: return best found if we're running out of time
    start_time = time.time()
    deadline = 85.0  # 85 seconds, leaving 5s safety margin before 90s kill
    
    # Step 1: Sieve to compute largest prime factor for all numbers up to N
    # LPF[i] will store the largest prime factor of i
    lpf = list(range(N + 1))
    lpf[0] = 0
    lpf[1] = 1
    
    # Sieve: for each prime p, update lpf for multiples of p
    for p in range(2, N + 1):
        if lpf[p] == p:  # p is prime
            for multiple in range(p, N + 1, p):
                lpf[multiple] = p
    
    # Step 2: Linear scan over all splits
    best_max_p = N  # worst case
    best_a = 1
    best_b = N - 1
    
    for a in range(1, N):
        b = N - a
        pa = lpf[a]
        pb = lpf[b]
        current_max = pa if pa > pb else pb
        if current_max < best_max_p:
            best_max_p = current_max
            best_a = a
            best_b = b
        
        # Deadline check
        if time.time() > deadline:
            break
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
