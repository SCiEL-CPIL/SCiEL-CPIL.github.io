import time
import numpy as np
from fractions import Fraction
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill

    N = 10_000_000

    # Sieve for sigma
    sigma = np.ones(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        if time.time() > deadline - 20:
            break
        sigma[d::d] += d

    # Build fibers: map sigma_value -> sorted list of indices
    # Use numpy argsort approach for efficiency
    # Only consider indices 2..N (index 1 has sigma=1, trivial)
    indices = np.arange(2, N + 1, dtype=np.int64)
    sig_vals = sigma[indices]

    # Sort indices by sigma value, then within same sigma by index
    # Use lexsort: primary key = sigma, secondary key = index
    order = np.lexsort((indices, sig_vals))
    sorted_indices = indices[order]
    sorted_sigs = sig_vals[order]

    # Now find pairs within each fiber using two-pointer
    best_err_num = None
    best_err_den = None
    best_a = None
    best_b = None

    # Iterate through fibers
    i = 0
    n = len(sorted_indices)
    while i < n:
        if time.time() > deadline - 5:
            break

        # Find extent of current fiber
        j = i
        while j < n and sorted_sigs[j] == sorted_sigs[i]:
            j += 1

        fiber_size = j - i
        if fiber_size >= 2:
            # Two-pointer: for each b in fiber, find a in fiber closest to 3b/2
            # The fiber is sorted by index, so we can use binary search
            fiber = sorted_indices[i:j]

            # For each b in fiber, the ideal a = 3b/2
            # We want to minimize |2a - 3b| over a in fiber
            # Since fiber is sorted, for each b we can binary search for 3b/2

            # Vectorized approach: for each b, compute target = 3b/2
            # Then use searchsorted to find candidate positions
            targets = (3 * fiber) / 2.0

            # For each target, the best a is at position floor(target) or ceil(target) in fiber
            # Use searchsorted
            pos = np.searchsorted(fiber, targets, side='left')

            # Candidate positions: pos-1, pos, pos+1 (clamped)
            for offset in [-1, 0, 1]:
                cand_pos = np.clip(pos + offset, 0, fiber_size - 1)
                a_vals = fiber[cand_pos]
                b_vals = fiber

                # Compute error: |2a - 3b| / (2b)
                # Only consider a != b
                mask = (a_vals != b_vals)
                if not np.any(mask):
                    continue

                a_use = a_vals[mask]
                b_use = b_vals[mask]

                err_num_arr = np.abs(2 * a_use - 3 * b_use)
                err_den_arr = 2 * b_use

                # Exclude zero error (invalid)
                nonzero = err_num_arr > 0
                if not np.any(nonzero):
                    continue

                err_num_arr = err_num_arr[nonzero]
                err_den_arr = err_den_arr[nonzero]
                a_use = a_use[nonzero]
                b_use = b_use[nonzero]

                if len(err_num_arr) == 0:
                    continue

                # Find minimum of err_num/err_den
                # This is not simply min of err_num or max of err_den
                # Need to compare fractions
                # For small fibers, do it element-wise
                # For large fibers, we can find the minimum by comparing

                # Simple approach: compute the ratio as float for ranking, then verify
                ratios = err_num_arr.astype(np.float64) / err_den_arr.astype(np.float64)
                min_idx = np.argmin(ratios)

                en = int(err_num_arr[min_idx])
                ed = int(err_den_arr[min_idx])
                ea = int(a_use[min_idx])
                eb = int(b_use[min_idx])

                if en == 0:
                    continue

                if best_err_num is None:
                    best_err_num = en
                    best_err_den = ed
                    best_a = ea
                    best_b = eb
                else:
                    # Compare en/ed vs best_err_num/best_err_den
                    lhs = en * best_err_den
                    rhs = best_err_num * ed
                    if lhs < rhs:
                        best_err_num = en
                        best_err_den = ed
                        best_a = ea
                        best_b = eb

        i = j

    if best_a is None:
        return [11, 6]

    return [int(best_a), int(best_b)]
