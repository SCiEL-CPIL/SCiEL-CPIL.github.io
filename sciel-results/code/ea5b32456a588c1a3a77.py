"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def construct():
    start_time = time.time()
    DEADLINE = start_time + 78  # safety margin before 90s kill
    N = 1000000
    MIN_N = 10000

    # 1. Sieve for Smallest Prime Factor (SPF)
    spf = list(range(N + 1))
    for i in range(2, isqrt(N) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # 2. Compute powerful part P(n) using multiplicative property and DP
    # P(n) is multiplicative.
    # For n = p^k * m where gcd(p, m) = 1:
    # P(n) = P(p^k) * P(m)
    # P(p^k) = p^k if k >= 2, else 1.
    
    # We compute P(n) for all n in [1, N] using the recurrence:
    # Let p = spf[n], m = n // p.
    # If v_p(m) >= 2 (i.e., m % (p*p) == 0): P(n) = P(m) * p
    # If v_p(m) == 1 (i.e., m % p == 0 and m % (p*p) != 0): P(n) = P(m) * p^2
    # If v_p(m) == 0 (i.e., m % p != 0): P(n) = P(m)
    
    P = [1] * (N + 1)
    # P[1] = 1 is already set
    
    for i in range(2, N + 1):
        p = spf[i]
        m = i // p
        pp = p * p
        if m % pp == 0:
            P[i] = P[m] * p
        elif m % p == 0:
            P[i] = P[m] * pp
        else:
            P[i] = P[m]

    # 3. Find n in [MIN_N, N-1] that maximizes P[n] * P[n+1]
    best_n = MIN_N
    best_val = 0
    
    # We only need to check up to N-1 because we need n+1 <= N
    for n in range(MIN_N, N):
        val = P[n] * P[n + 1]
        if val > best_val:
            best_val = val
            best_n = n
            
    return best_n

# EVOLVE-BLOCK-END
