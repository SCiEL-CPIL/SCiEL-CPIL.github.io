# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_N = 256
_DEADLINE = 85.0

def _max_abs_ap_sum(signs):
    """Compute max |sum| over all arithmetic progressions.
    For step d (1-indexed), positions r, r+d, r+2d,... (r=1..d) form contiguous
    subsequences. All AP sums are contiguous subarray sums in these sequences."""
    n = len(signs)
    max_disc = 1
    for d in range(1, n):
        for r in range(d):
            # 0-indexed positions: r, r+d, r+2d, ...
            seq = signs[r::d]
            L = len(seq)
            if L <= 1:
                continue
            # Compute max and min contiguous subarray sum
            csum = np.zeros(L + 1, dtype=np.int32)
            csum[1:] = np.cumsum(seq, dtype=np.int32)
            # Max subarray: max over j of (csum[j] - min_{i<j} csum[i])
            min_so_far = csum[0]
            max_sub = 0
            min_sub = 0
            for i in range(1, L + 1):
                v = csum[i]
                d1 = v - min_so_far
                if d1 > max_sub:
                    max_sub = d1
                d2 = min_so_far - v
                if d2 > min_sub:
                    min_sub = d2
                if v < min_so_far:
                    min_so_far = v
            abs_max = max(max_sub, min_sub)
            if abs_max > max_disc:
                max_disc = abs_max
    return max_disc

def _local_search(signs, deadline):
    """Local search to reduce max AP discrepancy."""
    n = _N
    signs = signs.copy()
    current = _max_abs_ap_sum(signs)
    best = current
    best_signs = signs.copy()
    
    start_time = time.time()
    while time.time() < deadline:
        pos = random.randint(0, n - 1)
        signs[pos] = -signs[pos]
        new_val = _max_abs_ap_sum(signs)
        if new_val < current:
            current = new_val
            if new_val < best:
                best = new_val
                best_signs = signs.copy()
        else:
            signs[pos] = -signs[pos]  # revert
    return best_signs, best

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    # Multiple random restarts with local search
    rng = random.Random(123)
    best_val = 10**9
    best_signs = None
    
    while time.time() < deadline - 3:
        # Random initialization
        signs = np.array([rng.choice((-1, 1)) for _ in range(_N)], dtype=np.int32)
        result_signs, result_val = _local_search(signs, deadline - 1)
        if result_val < best_val:
            best_val = result_val
            best_signs = result_signs
    
    if best_signs is None:
        best_signs = np.array([1] * _N, dtype=np.int32)
    
    return [int(s) for s in best_signs]
# EVOLVE-BLOCK-END
