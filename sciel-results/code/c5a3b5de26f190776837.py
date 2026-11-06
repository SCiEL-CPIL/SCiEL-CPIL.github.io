"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    N = 123457
    # Compute largest prime factor for each number up to N using sieve
    lpf = [0] * (N + 1)
    lpf[1] = 1
    for i in range(2, N + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i  # last assignment gives largest prime factor
    
    # Search for a in [1, N-1] minimizing max(lpf[a], lpf[N-a])
    best_val = float('inf')
    best_a = 1
    for a in range(1, N):
        b = N - a
        val = max(lpf[a], lpf[b])
        if val < best_val:
            best_val = val
            best_a = a
    
    return [best_a, N - best_a]
# EVOLVE-BLOCK-END
