"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    deadline = time.time() + 80.0  # 80s safety margin before 90s kill
    N_MIN = 10000
    N_MAX = 1000000
    max_n = N_MAX + 1  # 1000001

    # Build a simple prime sieve to get all primes up to max_n
    is_prime = [True] * (max_n + 1)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, int(max_n**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, max_n + 1, i):
                is_prime[j] = False

    # Sieve-based computation of powerful parts
    # pp[m] starts at 1 for all m. For each prime p and each power p^k (k>=2),
    # multiply pp[m] by p for every multiple m of p^k.
    pp = [1] * (max_n + 1)

    for p in range(2, max_n + 1):
        if not is_prime[p]:
            continue
        pk = p * p
        while pk <= max_n:
            for m in range(pk, max_n + 1, pk):
                pp[m] *= p
            pk *= p

    # Scan consecutive pairs to find the maximum product
    best_val = 0
    best_n = N_MIN

    for n in range(N_MIN, N_MAX + 1):
        if time.time() > deadline:
            break
        val = pp[n] * pp[n + 1]
        if val > best_val:
            best_val = val
            best_n = n

    return best_n
# EVOLVE-BLOCK-END
