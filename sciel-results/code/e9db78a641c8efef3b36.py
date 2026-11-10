import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin

    N = 1000000
    best_n = 10000

    pp = np.ones(N + 2, dtype=np.int64)

    limit = int(math.isqrt(N)) + 1
    is_prime = bytearray([1]) * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            step = i
            start = i * i
            is_prime[start:limit+1:step] = bytearray(len(range(start, limit+1, step)))

    primes = [i for i in range(2, limit + 1) if is_prime[i]]

    for p in primes:
        p2 = p * p
        if p2 > N + 1:
            continue

        pe = p2
        while pe <= N + 1:
            pe_next = pe * p
            if pe_next > N + 1:
                # Final (largest) prime power for this prime: vectorized
                indices = np.arange(pe, N + 2, pe)
                pp[indices] *= pe
            else:
                # k not divisible by p: use mask-based vectorized approach
                max_k = (N + 1) // pe
                # Build all multiples of pe up to N+1
                all_indices = np.arange(pe, N + 2, pe)
                # k values: all_indices // pe
                k_vals = all_indices // pe
                # Mask: k not divisible by p
                mask = (k_vals % p) != 0
                pp[all_indices[mask]] *= pe
            pe = pe_next
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    # Vectorized search for max product
    if time.time() < deadline - 2:
        prods = pp[10000:N+1].astype(np.int64) * pp[10001:N+2].astype(np.int64)
        idx = int(np.argmax(prods))
        best_n = 10000 + idx
    else:
        max_prod = 0
        for n in range(10000, N + 1):
            prod = int(pp[n]) * int(pp[n+1])
            if prod > max_prod:
                max_prod = prod
                best_n = n

    return best_n
