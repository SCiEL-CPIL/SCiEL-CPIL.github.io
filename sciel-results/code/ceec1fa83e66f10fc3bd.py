import time
import math
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 150s kill

    best_M = 333312
    best_ks = [336, 372, 434]

    def update_best(M, ks):
        nonlocal best_M, best_ks
        if isinstance(ks, np.ndarray):
            ks_list = [int(x) for x in ks]
        else:
            ks_list = list(ks)
        if len(ks_list) > len(best_ks):
            best_M = int(M)
            best_ks = ks_list

    def smooth_search(max_k, small_primes):
        """Generate all k <= max_k composed of small_primes, compute k*sigma(k), group by value."""
        local_start = time.time()
        try:
            # Generate all smooth numbers <= max_k using BFS/DFS over prime exponents
            # For each smooth number k, compute sigma(k) from its factorization
            # sigma(k) = product of (p^(a+1)-1)/(p-1) for k = prod p^a

            groups = defaultdict(list)  # M -> list of k values

            # Use a recursive approach to enumerate smooth numbers
            # Primes sorted
            n_primes = len(small_primes)

            def gen_smooth(idx, current_k, current_sigma):
                """Recursively generate smooth numbers starting from prime index idx."""
                if time.time() > deadline - 20:
                    return

                if current_k > max_k:
                    return

                # Record this smooth number
                if current_k >= 1:
                    M = current_k * current_sigma
                    if M <= 10**18:
                        # Check k <= isqrt(M): k^2 <= k*sigma(k) => k <= sigma(k)
                        if current_k <= current_sigma:
                            groups[M].append(current_k)

                if idx >= n_primes:
                    return

                p = small_primes[idx]
                # Try all powers of p
                pk = 1
                sigma_p = 1  # sigma(p^0) = 1
                while pk * p <= max_k // max(1, current_k):
                    pk *= p
                    sigma_p = sigma_p * p + 1  # sigma(p^a) = 1 + p + ... + p^a
                    new_k = current_k * pk
                    if new_k > max_k:
                        break
                    new_sigma = current_sigma * sigma_p
                    if new_k * new_sigma > 10**18:
                        break
                    gen_smooth(idx + 1, new_k, new_sigma)

            gen_smooth(0, 1, 1)

            if time.time() > deadline - 10:
                return

            # Find the M with the most k values
            if groups:
                max_count = 0
                max_M = 0
                max_ks = []
                for M, ks in groups.items():
                    if len(ks) > max_count:
                        max_count = len(ks)
                        max_M = M
                        max_ks = ks

                if max_count > len(best_ks):
                    # Verify all k_i satisfy k_i <= isqrt(M)
                    isqrt_M = math.isqrt(max_M)
                    valid_ks = [int(x) for x in max_ks if int(x) <= isqrt_M]
                    if len(valid_ks) > len(best_ks):
                        update_best(max_M, valid_ks)

                print(f"Smooth search: max_k={max_k}, primes={len(small_primes)}, "
                      f"groups={len(groups)}, best_count={max_count}, "
                      f"elapsed={time.time()-local_start:.1f}s")

        except Exception as e:
            print(f"Error in smooth search: {e}")

    # Define small primes to use for smooth number generation
    # Primes less than 50: 2,3,5,7,11,13,17,19,23,29,31,37,41,43,47
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

    # Try different max_k values, starting with smaller ones to ensure we get results
    # With 15 primes, the number of smooth numbers grows rapidly
    # For max_k = 10^6, we should have manageable counts
    # For max_k = 10^7, more candidates but still tractable

    max_k_values = [1000000, 5000000, 10000000]

    for max_k in max_k_values:
        if time.time() > deadline - 30:
            break
        smooth_search(max_k, small_primes)
        if time.time() > deadline - 20:
            break

    # Also try with fewer primes but larger max_k
    # With fewer primes, we can go to larger max_k
    small_primes_10 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    small_primes_7 = [2, 3, 5, 7, 11, 13, 17]

    if time.time() < deadline - 30:
        smooth_search(50000000, small_primes_10)

    if time.time() < deadline - 20:
        smooth_search(100000000, small_primes_7)

    # Ensure types are correct
    best_M = int(best_M)
    best_ks = [int(x) for x in best_ks]

    return {'M': best_M, 'k': best_ks}
