import time
import numpy as np
from math import isqrt

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # Safety margin before 150s kill

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

    def compute_phi_sieve_numpy(limit):
        """Compute phi for 0..limit using NumPy."""
        phi = np.arange(limit + 1, dtype=np.int64)
        for i in range(2, limit + 1):
            if phi[i] == i:  # i is prime
                indices = np.arange(i, limit + 1, i)
                phi[indices] -= phi[indices] // i
        return phi

    def find_longest_distinct_run(phi_vals):
        """
        phi_vals: array where phi_vals[i] = phi(i)
        Finds max length k and start s (1-indexed) such that phi(s)..phi(s+k-1) are distinct.
        """
        n = len(phi_vals)
        if n < 2:
            return 1, 1

        last_seen = {}
        start = 1  # 1-indexed start of current window
        max_len = 0
        max_start = 1

        # Iterate from 1 to n-1 (indices in phi_vals)
        # phi_vals[0] is phi(0), we ignore it for s>=1
        for i in range(1, n):
            val = phi_vals[i]
            if val in last_seen and last_seen[val] >= start:
                start = last_seen[val] + 1
            last_seen[val] = i

            current_len = i - start + 1
            if current_len > max_len:
                max_len = current_len
                max_start = start

        return max_len, max_start

    best_k = 1
    best_s = 1

    # Try increasing limits
    # 10^7 is very fast. 10^8 is moderate. 2*10^8 is risky but worth trying.
    limits = [10_000_000, 50_000_000, 100_000_000, 200_000_000]

    for limit in limits:
        if time_remaining() < 20.0:  # Need time for sliding window too
            break

        try:
            phi_vals = compute_phi_sieve_numpy(limit)
            k, s = find_longest_distinct_run(phi_vals)

            if k > 200:
                k = 200
                # If k > 200, we can just take the first 200 of the run
                # The start s found is the start of the max run.
                # We need s + k - 1 <= limit.
                # If the run is longer than 200, s is valid for length 200.

            if k > best_k:
                best_k = k
                best_s = s
        except Exception:
            # Memory error or other
            break

    return {'start': best_s, 'length': best_k}
