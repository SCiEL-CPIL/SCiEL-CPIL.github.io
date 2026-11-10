"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    N = 123457
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin before 60s limit

    # Precompute largest prime factor for all numbers up to N
    lpf = [0] * (N + 1)
    lpf[1] = 1
    
    for p in range(2, N + 1):
        if lpf[p] == 0:  # p is prime
            for j in range(p, N + 1, p):
                lpf[j] = p
        
        # Check time periodically, though this loop is very fast
        if p % 10000 == 0 and time.time() > deadline:
            # Fallback to a simple heuristic if somehow too slow (unlikely)
            # But for N=123457, this is instant.
            pass

    best_a = 1
    best_b = N - 1
    min_max_p = float('inf')
    
    for a in range(1, N):
        b = N - a
        max_p = lpf[a] if lpf[a] > lpf[b] else lpf[b]
        if max_p < min_max_p:
            min_max_p = max_p
            best_a = a
            best_b = b
            
            if min_max_p == 2: # Theoretical minimum for even numbers, but N is odd so one is odd
                # If N is odd, one of a,b is even, one is odd.
                # If the even one is a power of 2, P(even)=2.
                # If the odd one is 1, P(odd)=1.
                # But a,b >= 1.
                # Minimum possible max P is likely small.
                # We can break early if we find a very small value, but let's finish the scan to be sure.
                # Actually, if we find P(a)=2 and P(b)=1 (impossible as b>1 if a<N-1? No, b can be 1)
                # If a = N-1, b=1. P(1)=1. P(N-1) = ?
                # N = 123457 is odd.
                # Let's just finish the loop, it's fast.
                pass

    return [best_a, best_b]
# EVOLVE-BLOCK-END
