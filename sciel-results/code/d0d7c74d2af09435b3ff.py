import math
import time
import numpy as np

def construct():
    N = 1000000
    MIN_N = 10000
    deadline = time.time() + 80  # Safety margin for 90s limit

    # 1. Sieve for Smallest Prime Factor (SPF)
    spf = np.arange(N + 1, dtype=np.int64)
    for i in range(2, int(math.isqrt(N)) + 1):
        if spf[i] == i:
            for j in range(i * i, N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # 2. Precompute powerful part for all numbers up to N
    # Use efficient iterative approach
    pp = np.ones(N + 1, dtype=np.int64)
    spf_local = spf
    pp_local = pp

    # Process each number by its prime factorization
    for p in range(2, N + 1):
        if spf_local[p] != p:  # not prime, skip
            continue
        # p is prime, find all multiples and compute p^e contribution
        for e in range(2, 20):  # p^e <= N
            pe = p ** e
            if pe > N:
                break
            # For each multiple of pe, we need to check if the exponent of p is exactly e or higher
            # and multiply by p^e if exponent >= 2
            for k in range(pe, N + 1, pe):
                # Check if k is divisible by pe but not by p^(e+1)
                # If divisible by p^(e+1), we'll handle it when we process that higher power
                if k % (pe * p) == 0:
                    continue  # Will be handled at higher exponent
                # k has exactly p^e in its factorization (for prime p)
                if e >= 2:
                    pp_local[k] *= pe

    # 3. Find max product of consecutive powerful parts
    best_n = MIN_N
    best_val = 0
    pp_local = pp

    # Vectorized search for maximum product
    products = pp_local[MIN_N:N] * pp_local[MIN_N + 1:N + 1]
    max_idx = np.argmax(products)
    best_val = products[max_idx]
    best_n = MIN_N + max_idx

    # 4. CRT-based Constructive Search to find potentially better candidates
    # Generate powerful numbers up to some limit
    powerful_nums = []
    for x in range(2, 5000):
        if pp_local[x] == x:
            powerful_nums.append(x)

    # Sort by value descending to try larger A, B first
    powerful_nums.sort(reverse=True)

    # Check CRT candidates
    checked_crt = set()
    best_crt_val = 0
    best_crt_n = best_n

    for A in powerful_nums[:50]:
        if time.time() > deadline:
            break
        for B in powerful_nums[:50]:
            if A == B:
                continue
            if math.gcd(A, B) != 1:
                continue

            try:
                inv_A = pow(A, -1, B)
            except ValueError:
                continue

            k0 = (-inv_A) % B
            n0 = k0 * A
            L = A * B

            if L == 0:
                continue

            if n0 < MIN_N:
                t = (MIN_N - n0 + L - 1) // L
            else:
                t = 0

            n_candidate = n0 + t * L

            if n_candidate <= N and n_candidate >= MIN_N:
                if n_candidate not in checked_crt:
                    checked_crt.add(n_candidate)
                    val = pp_local[n_candidate] * pp_local[n_candidate + 1]
                    if val > best_val:
                        best_val = val
                        best_n = n_candidate
                    if val > best_crt_val:
                        best_crt_val = val
                        best_crt_n = n_candidate

    # 5. Local search around the best candidate
    search_range = 1000
    start = max(MIN_N, best_n - search_range)
    end = min(N, best_n + search_range)

    for n in range(start, end + 1):
        if time.time() > deadline:
            break
        val = pp_local[n] * pp_local[n + 1]
        if val > best_val:
            best_val = val
            best_n = n

    return best_n
