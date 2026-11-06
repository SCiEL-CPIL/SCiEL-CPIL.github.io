"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START

def largest_prime_factor(n):
    """Compute the largest prime factor of n by exact trial division."""
    if n < 2:
        return 1
    lpf = 1
    while n % 2 == 0:
        lpf = 2
        n //= 2
    f = 3
    while f * f <= n:
        while n % f == 0:
            lpf = f
            n //= f
        f += 2
    if n > 1:
        lpf = n
    return lpf

def construct():
    """
    Find n in [10000, 1000000] that minimizes the largest prime factor of n(n+1).
    Uses an optimized sieve to compute LPF for all integers up to 1000001.
    """
    start_time = time.time()
    deadline = start_time + 80
    
    N = 1000001  # need lpf up to 1000001
    
    # Initialize lpf array
    lpf = [0] * (N + 1)
    
    # Optimized sieve: only process primes, update multiples
    for i in range(2, N + 1):
        if lpf[i] == 0:  # i is prime
            for j in range(i, N + 1, i):
                lpf[j] = i
    
    if time.time() > deadline:
        return 10000
    
    # Scan for the minimum max(lpf[n], lpf[n+1])
    best_n = 10000
    best_val = float('inf')
    
    for n in range(10000, N):
        val = max(lpf[n], lpf[n + 1])
        if val < best_val:
            best_val = val
            best_n = n
            # Early termination: if we find a very small value, we can stop
            # The smallest possible LPF for n>=10000 is at least 2
            # If best_val == 2, that's optimal (but unlikely for n>=10000)
            if best_val <= 7:
                # Check if we can do better; 2 is the absolute minimum
                # For n>=10000, both n and n+1 being powers of 2 is impossible
                # (consecutive powers of 2 don't exist beyond 1,2)
                # So 3 might be achievable if one is power of 2 and other power of 3
                # But let's keep scanning a bit more to be safe
                pass
    
    # Verify with exact trial division
    if time.time() < deadline - 5:
        verified_val = max(largest_prime_factor(best_n), largest_prime_factor(best_n + 1))
    
    return best_n

# EVOLVE-BLOCK-END
