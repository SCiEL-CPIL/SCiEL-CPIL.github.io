"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    """
    Find n in [10000, 1000000] that minimizes the largest prime factor of n(n+1).
    
    Strategy:
    1. Precompute the largest prime factor (LPF) for all numbers up to 1,000,001
       using a sieve-like approach.
    2. Scan all valid n in [10000, 999999] (since n+1 <= 1000000 is not required, 
       but n+1 can be up to 1000000 if n=999999, so we need LPF up to 1000000).
       Wait, n is in [10000, 1000000]. So n+1 is in [10001, 1000001].
       We need LPF for numbers up to 1,000,001.
    3. For each n, compute max(LPF[n], LPF[n+1]).
    4. Track the n with the minimum such value.
    5. Return that n.
    
    Deadline guard: If precomputation or scan takes too long, return best found so far.
    """
    start_time = time.time()
    deadline = start_time + 85  # 5 second safety margin before 90s kill
    
    # We need LPF for numbers up to 1,000,001
    MAX_VAL = 1000001
    
    # Initialize LPF array
    lpf = [0] * (MAX_VAL + 1)
    
    # Sieve to compute largest prime factor
    # For each prime p, mark multiples of p with p as their LPF
    # We iterate p from 2 to MAX_VAL
    # Optimization: only need to go up to sqrt(MAX_VAL) for the sieve part? 
    # No, standard LPF sieve: 
    # for i in range(2, MAX_VAL+1):
    #   if lpf[i] == 0: # i is prime
    #     for j in range(i, MAX_VAL+1, i):
    #       lpf[j] = i
    # This is O(N log log N) roughly.
    
    for i in range(2, MAX_VAL + 1):
        if lpf[i] == 0:
            # i is prime
            for j in range(i, MAX_VAL + 1, i):
                lpf[j] = i
        if time.time() > deadline:
            # Fallback if sieve takes too long (unlikely, but safe)
            return 10000
    
    # Now scan for best n
    best_n = 10000
    best_lpf_val = float('inf')
    
    # n ranges from 10000 to 1000000
    # We need lpf[n] and lpf[n+1]
    # n+1 goes up to 1000001, which is within MAX_VAL
    
    for n in range(10000, 1000001):
        current_max_lpf = max(lpf[n], lpf[n + 1])
        if current_max_lpf < best_lpf_val:
            best_lpf_val = current_max_lpf
            best_n = n
        if time.time() > deadline:
            break
    
    return best_n

# EVOLVE-BLOCK-END
