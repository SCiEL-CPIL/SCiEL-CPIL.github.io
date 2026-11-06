"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math

def construct():
    MAX_N = 1000000
    MIN_N = 10000
    
    # We need spf up to MAX_N + 1
    limit = MAX_N + 1
    spf = list(range(limit + 1))
    
    # Sieve of Eratosthenes to find Smallest Prime Factor
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i

    def get_lpf(x):
        """Returns the largest prime factor of x using precomputed spf."""
        lpf = 1
        while x > 1:
            p = spf[x]
            lpf = p
            x //= p
        return lpf

    best_n = MIN_N
    best_val = float('inf')
    
    # Iterate through the range
    for n in range(MIN_N, MAX_N + 1):
        lpf_n = get_lpf(n)
        lpf_np1 = get_lpf(n + 1)
        current_val = max(lpf_n, lpf_np1)
        
        if current_val < best_val:
            best_val = current_val
            best_n = n
            # Optional: Early termination if best_val is very small (e.g., 1 is impossible, 2 implies both are powers of 2, which can't be consecutive > 1)
            # The smallest possible value for max(lpf(n), lpf(n+1)) for n>=10000 is likely small.
            # If we find a value that is theoretically minimal, we could stop, but let's just finish the loop as it's fast enough.
            
    return best_n
# EVOLVE-BLOCK-END
