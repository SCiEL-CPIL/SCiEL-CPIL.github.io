"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np
from collections import defaultdict

def _sigma_sieve_np(n):
    """Compute sigma(k) for k=1..n using NumPy vectorized sieve."""
    sigma = np.zeros(n + 1, dtype=np.int64)
    for d in range(1, n + 1):
        sigma[d::d] += d
    return sigma

def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150s kill

    # Start with the known 3-way collision as fallback
    best_M = 333312
    best_k = [336, 372, 434]
    best_count = 3

    max_M = 10**18

    # Try increasing limits with NumPy-accelerated sieve
    for limit in [5_000_000, 10_000_000, 20_000_000, 40_000_000, 80_000_000, 150_000_000]:
        if time.time() > deadline - 30:
            break
        try:
            sigma = _sigma_sieve_np(limit)
            # Compute v = k * sigma(k) for all k
            ks_arr = np.arange(1, limit + 1, dtype=np.int64)
            vals = ks_arr * sigma[1:]

            # Filter: v <= max_M and k <= sigma(k) (always true but keep for safety)
            mask = (vals <= max_M) & (ks_arr <= sigma[1:])
            valid_ks = ks_arr[mask]
            valid_vals = vals[mask]

            if len(valid_ks) == 0:
                del sigma, vals, mask, valid_ks, valid_vals
                continue

            # Group by value using argsort for efficiency
            order = np.argsort(valid_vals, kind='stable')
            sorted_vals = valid_vals[order]
            sorted_ks = valid_ks[order]

            # Find groups of equal values
            # Use diff to find boundaries
            changes = np.concatenate(([True], sorted_vals[1:] != sorted_vals[:-1]))
            group_starts = np.where(changes)[0]
            group_ends = np.concatenate((group_starts[1:], [len(sorted_ks)]))

            # Find the group with the most elements
            group_sizes = group_ends - group_starts
            max_idx = np.argmax(group_sizes)
            max_size = group_sizes[max_idx]

            if max_size > best_count:
                start = group_starts[max_idx]
                end = group_ends[max_idx]
                group_ks = sorted_ks[start:end]
                group_v = int(sorted_vals[start])
                # Verify k <= isqrt(M)
                sq = math.isqrt(group_v)
                valid_group = [int(k) for k in group_ks if k <= sq]
                if len(valid_group) > best_count:
                    best_count = len(valid_group)
                    best_M = group_v
                    best_k = valid_group[:1000]

            del sigma, vals, mask, valid_ks, valid_vals, order, sorted_vals, sorted_ks
            del changes, group_starts, group_ends, group_sizes, max_idx, max_size

        except (MemoryError, Exception):
            break

    # Ensure types are correct
    best_M = int(best_M)
    best_k = [int(k) for k in best_k[:1000]]

    return {'M': best_M, 'k': best_k}
# EVOLVE-BLOCK-END
