import time
import math
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill

    N_MIN = 10000
    N_MAX = 1000000
    M = N_MAX + 1  # we need lpf for values up to 1000001

    # Sieve of Eratosthenes using numpy
    is_prime = np.ones(M + 1, dtype=bool)
    is_prime[0] = False
    is_prime[1] = False
    for i in range(2, math.isqrt(M) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = False

    # Collect primes
    primes = np.array([i for i in range(2, M + 1) if is_prime[i]])

    # Compute largest prime factor for each number up to M using numpy
    # lpf[x] = largest prime factor of x
    lpf = np.zeros(M + 1, dtype=np.int32)
    for p in primes:
        multiples = np.arange(p, M + 1, p)
        lpf[multiples] = p

    # Find n in [N_MIN, N_MAX] minimizing max(lpf[n], lpf[n+1])
    best_n = N_MIN
    best_val = float('inf')
    for n in range(N_MIN, N_MAX + 1):
        val = max(lpf[n], lpf[n+1])
        if val < best_val:
            best_val = val
            best_n = n
        if time.time() - start_time > DEADLINE:
            break

    return int(best_n)
