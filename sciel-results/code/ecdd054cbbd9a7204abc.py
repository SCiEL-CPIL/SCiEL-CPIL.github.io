"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Return two positive integers a, b with a+b=123457.
    Minimize max(P(a), P(b)), where P is the largest prime factor and P(1)=1.
    """
    N = 123457
    
    # Deadline guard: return best found if we're running out of time
    deadline = time.time() + 50.0  # 50s safety margin within 60s limit
    
    # Step 1: Sieve to compute the largest prime factor for all numbers up to N
    # largest_prime_factor[i] = largest prime factor of i
    lp = list(range(N + 1))  # lp[i] will hold the largest prime factor of i
    
    # Standard sieve: for each prime p, mark its multiples
    for i in range(2, N + 1):
        if lp[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                lp[j] = i  # i is a prime factor of j; since we go in increasing order,
                           # the last prime that divides j is the largest
    
    # Step 2: Linear scan over all splits a=1..N-1, b=N-a
    best_a = 1
    best_b = N - 1
    best_val = max(lp[1], lp[N - 1])
    
    for a in range(1, N):
        b = N - a
        val = max(lp[a], lp[b])
        if val < best_val:
            best_val = val
            best_a = a
            best_b = b
            # Early termination: if we hit 1, that's the absolute minimum possible
            if best_val == 1:
                break
        # Time guard
        if time.time() > deadline:
            break
    
    return [best_a, best_b]
# EVOLVE-BLOCK-END
