"""Search for M with many distinct k where k*sigma(k)=M, k<=isqrt(M)."""
import math
import time
from collections import defaultdict
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 40s safety margin

    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3

    # Strategy: Compute sigma(k) for k up to K using numpy, group k*sigma(k),
    # find M with most preimages. Also try a "smooth number" focused search
    # for higher density collisions.

    def sigma_sieve_numpy(K):
        """Compute sigma(1..K) using numpy vectorized sieve."""
        sigma = np.zeros(K + 1, dtype=np.int64)
        for d in range(1, K + 1):
            sigma[d::d] += d
        return sigma

    def find_best_from_K(K, deadline, best_count, best_M, best_k):
        """Given K, compute sigma, group, return updated best."""
        if time.time() > deadline - 20:
            return best_M, best_k, best_count

        sigma = sigma_sieve_numpy(K)

        if time.time() > deadline - 10:
            return best_M, best_k, best_count

        ks_range = np.arange(1, K + 1, dtype=np.int64)
        Ms = ks_range * sigma[1:]

        # Only consider M <= 10^18
        valid_mask = Ms <= 10**18
        Ms_valid = Ms[valid_mask]
        ks_valid = ks_range[valid_mask]

        if len(Ms_valid) == 0:
            return best_M, best_k, best_count

        # Group by M
        unique_Ms, inverse, counts = np.unique(Ms_valid, return_inverse=True, return_counts=True)

        max_count_idx = np.argmax(counts)
        max_count = int(counts[max_count_idx])

        if max_count > best_count:
            best_M = int(unique_Ms[max_count_idx])
            # Find all k with this M
            k_indices = np.where(Ms_valid == best_M)[0]
            best_k = [int(x) for x in ks_valid[k_indices]]
            best_count = max_count

        return best_M, best_k, best_count

    try:
        # Main search: K = 15,000,000 (balance of memory and time)
        K_main = 15000000
        best_M, best_k, best_count = find_best_from_K(
            K_main, deadline, best_count, best_M, best_k
        )
    except MemoryError:
        try:
            K_main = 8000000
            best_M, best_k, best_count = find_best_from_K(
                K_main, deadline, best_count, best_M, best_k
            )
        except MemoryError:
            K_main = 4000000
            best_M, best_k, best_count = find_best_from_K(
                K_main, deadline, best_count, best_M, best_k
            )

    # Secondary: smooth-number focused search
    # Generate k composed of small primes only, where sigma(k) grows slowly
    # relative to k, enabling more collisions for the same M.
    if time.time() < deadline - 30:
        try:
            # Primes to use for smooth numbers
            small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
            # Limit exponents to keep k <= 10^7
            # For each prime p, max exponent e such that p^e <= 10^7
            max_K_smooth = 10000000
            max_exponents = []
            for p in small_primes:
                e = 0
                val = 1
                while val * p <= max_K_smooth:
                    val *= p
                    e += 1
                max_exponents.append(e)

            # Generate all smooth numbers <= max_K_smooth
            # Using recursive generation
            smooth_nums = []

            def gen_smooth(idx, current):
                if time.time() > deadline - 20:
                    return
                if current > max_K_smooth:
                    return
                if idx == len(small_primes):
                    if current > 1:
                        smooth_nums.append(current)
                    return
                p = small_primes[idx]
                max_e = max_exponents[idx]
                for e in range(max_e + 1):
                    if current * (p ** e) > max_K_smooth:
                        break
                    gen_smooth(idx + 1, current * (p ** e))

            gen_smooth(0, 1)

            if len(smooth_nums) > 0:
                # Compute sigma for these smooth numbers
                # For smooth numbers, sigma can be computed from prime factorization
                # But easier: use a dictionary-based sigma computation
                sigma_dict = {}
                for k in smooth_nums:
                    if k in sigma_dict:
                        continue
                    # Compute sigma(k) by factoring (k is smooth, so fast)
                    n = k
                    sigma_val = 1
                    for p in small_primes:
                        if n % p == 0:
                            pe = 1
                            p_term = 1
                            while n % p == 0:
                                n //= p
                                pe *= p
                                p_term += pe
                            sigma_val *= p_term
                    if n > 1:
                        sigma_val *= (1 + n)
                    sigma_dict[k] = sigma_val

                # Group by M = k * sigma(k)
                groups = defaultdict(list)
                for k in smooth_nums:
                    M = k * sigma_dict[k]
                    if M <= 10**18:
                        groups[M].append(k)

                for M, ks in groups.items():
                    cnt = len(ks)
                    if cnt > best_count:
                        best_count = cnt
                        best_M = M
                        best_k = ks

        except Exception:
            pass  # Keep existing best

    # Ensure types are plain Python int and list
    best_k = [int(x) for x in best_k]
    best_M = int(best_M)

    return {'M': best_M, 'k': best_k}
