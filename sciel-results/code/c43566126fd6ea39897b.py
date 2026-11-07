"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin

    max_m = 10**18

    # Strategy: Generate k values where k*sigma(k) <= max_m
    # Find M with the most such k values
    # Use efficient generation with good pruning

    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

    m_to_ks = defaultdict(list)
    count = [0]

    def gen_smooth(idx, k, sigma_k):
        if time.time() > deadline:
            return

        if idx == len(primes):
            M = k * sigma_k
            if M <= max_m:
                m_to_ks[M].append(k)
                count[0] += 1
            return

        p = primes[idx]
        p_pow = 1
        sigma_p = 1
        for e in range(0, 35):
            new_k = k * p_pow
            new_sigma = sigma_k * sigma_p
            M = new_k * new_sigma

            if M > max_m:
                break

            gen_smooth(idx + 1, new_k, new_sigma)

            p_pow *= p
            if p_pow > max_m:
                break
            sigma_p += p_pow

    gen_smooth(0, 1, 1)

    # Find the best M
    best_m = 0
    best_count = 0
    for M, ks in m_to_ks.items():
        if len(ks) > best_count:
            best_count = len(ks)
            best_m = M

    if best_count == 0:
        return {'M': 333312, 'k': [336, 372, 434]}

    best_ks = m_to_ks[best_m]
    # Ensure distinct and within bounds
    best_ks = list(set(best_ks))
    sqrt_m = math.isqrt(best_m)
    best_ks = [k for k in best_ks if k <= sqrt_m]
    best_ks.sort()

    if not best_ks:
        return {'M': 333312, 'k': [336, 372, 434]}

    return {'M': int(best_m), 'k': [int(k) for k in best_ks]}
# EVOLVE-BLOCK-END
