"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    N = 123457
    
    # Pre-compute the largest prime factor (LPF) for all numbers up to N
    # lpf[x] will store the largest prime factor of x
    lpf = [0] * (N + 1)
    
    # Sieve to find largest prime factor
    # Iterating i from 2 to N, if lpf[i] is 0, i is prime.
    # For each prime i, update lpf for all multiples of i to i.
    # Since we iterate i in increasing order, the last update for a number j
    # will come from its largest prime factor.
    for i in range(2, N + 1):
        if lpf[i] == 0:
            for j in range(i, N + 1, i):
                lpf[j] = i
                
    # P(1) = 1
    lpf[1] = 1
    
    min_cost = float('inf')
    best_a = 1
    best_b = N - 1
    
    # Iterate through all possible a, b pairs
    # We only need to check up to N//2 due to symmetry
    for a in range(1, N // 2 + 1):
        b = N - a
        cost = max(lpf[a], lpf[b])
        if cost < min_cost:
            min_cost = cost
            best_a = a
            best_b = b
            
    return [best_a, best_b]
# EVOLVE-BLOCK-END
