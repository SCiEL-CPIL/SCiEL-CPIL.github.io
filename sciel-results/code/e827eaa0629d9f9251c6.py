"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_N = 256
_DEADLINE = 85.0  # seconds, safety margin before 90s kill

def _thue_morse_init():
    """Generate Thue-Morse sequence as initial low-discrepancy starting point."""
    signs = np.ones(_N, dtype=np.int32)
    for i in range(1, _N):
        # Thue-Morse: parity of number of 1s in binary representation
        t = i
        parity = 0
        while t:
            parity ^= (t & 1)
            t >>= 1
        signs[i] = -1 if parity else 1
    return signs

def _max_abs_ap_sum(signs):
    """
    Compute the maximum absolute sum over all arithmetic progressions.
    Uses prefix sums per (step, residue) class for efficient evaluation.
    """
    n = _N
    max_disc = 1  # singletons always give |sum| = 1
    
    for d in range(2, n):
        for r in range(d):
            # Extract subsequence at positions r, r+d, r+2d, ... (0-indexed)
            seq = signs[r:n:d]
            if len(seq) < 2:
                continue
            csum = np.cumsum(seq)
            # All contiguous subarray sums can be found via csum[j] - csum[i-1]
            # We need max |csum[j] - csum[i-1]| for 0 <= i-1 < j <= len(seq)
            # This is equivalent to max(csum) - min(csum) over the full csum array
            # but we need to be careful: subarray sum from i to j-1 is csum[j-1] - (csum[i-1] if i>0 else 0)
            # Let's use the standard trick: prepend 0 to csum
            csum_with_zero = np.concatenate(([0], csum))
            max_sub = csum_with_zero.max() - csum_with_zero.min()
            if max_sub > max_disc:
                max_disc = max_sub
            # Also check negative direction (already covered by max-min)
    
    return max_disc

def _local_search(signs, deadline):
    """
    Local search: flip one sign at a time if it reduces the max absolute AP sum.
    Uses incremental update where possible, but for simplicity we recompute
    the affected APs. For efficiency, we precompute which APs each position belongs to.
    """
    start_time = time.time()
    best_signs = signs.copy()
    best_score = _max_abs_ap_sum(best_signs)
    
    # Precompute for each position i, the list of (d, r, idx_in_seq) where position i appears
    # This is expensive to store fully, so we use a different approach:
    # For each position, determine which (d, r) pairs it participates in,
    # and its index within that subsequence.
    
    # Alternative: just do greedy local search with full recomputation but
    # limit the number of iterations based on time.
    
    n = _N
    improved = True
    while improved and (time.time() - start_time) < deadline - 5.0:
        improved = False
        for i in range(n):
            if (time.time() - start_time) > deadline - 5.0:
                break
            # Try flipping position i
            best_signs[i] *= -1
            new_score = _max_abs_ap_sum(best_signs)
            if new_score < best_score:
                best_score = new_score
                improved = True
            else:
                best_signs[i] *= -1  # revert
    
    return best_signs, best_score

def construct():
    start_time = time.time()
    
    # Start with Thue-Morse initialization
    signs = _thue_morse_init()
    score = _max_abs_ap_sum(signs)
    
    # Local search
    signs, score = _local_search(signs, _DEADLINE)
    
    # Convert to list of Python integers
    return [int(s) for s in signs]
# EVOLVE-BLOCK-END
