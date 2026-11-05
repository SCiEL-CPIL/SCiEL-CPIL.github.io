import math
import time

def construct():
    """
    Find a, b such that a + b = 123457 and max(P(a), P(b)) is minimized.
    P(n) is the largest prime factor of n, P(1)=1.
    """
    TARGET = 123457
    start_time = time.time()
    DEADLINE = 85  # safety margin below 90s kill

    # Precompute largest prime factor for all numbers up to TARGET
    lp = [0] * (TARGET + 1)
    lp[1] = 1

    # Sieve: for each prime p, mark multiples with p as largest prime factor
    # Since we go in increasing order, later (larger) primes overwrite,
    # so lp[n] ends up as the largest prime factor of n.
    for p in range(2, TARGET + 1):
        if lp[p] == 0:  # p is prime
            for multiple in range(p, TARGET + 1, p):
                lp[multiple] = p

    # Now find the minimum of max(lp[a], lp[b]) where a + b = TARGET, a,b >= 1
    best_max_pf = float('inf')
    best_a = 1
    best_b = TARGET - 1

    for a in range(1, TARGET):
        b = TARGET - a
        max_pf = lp[a] if lp[a] > lp[b] else lp[b]
        if max_pf < best_max_pf:
            best_max_pf = max_pf
            best_a = a
            best_b = b
            # Check deadline periodically
            if a % 10000 == 0 and time.time() - start_time > DEADLINE:
                break
            # Early exit: can't do better than 2 unless one is 1
            if best_max_pf <= 2:
                break

    if time.time() - start_time > DEADLINE:
        pass  # already have best

    return [best_a, best_b]
