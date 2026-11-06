import time
import numpy as np

N = 17
MIN_DIST = 6
MAX_WORD = 1 << N


def _build_conflict_matrix():
    """
    Build a bit-parallel conflict matrix.
    conflict[i] is a bitset (Python int) where bit j is 1 iff
    popcount(i ^ j) < MIN_DIST.
    """
    # Precompute popcount for all 17-bit values
    pc = np.zeros(MAX_WORD, dtype=np.uint8)
    for i in range(1, MAX_WORD):
        pc[i] = pc[i >> 1] + (i & 1)

    # Build conflict bitsets
    conflict = [0] * MAX_WORD
    # Vectorized: for each i, compute pc[i ^ j] for all j
    # This is expensive memory-wise (2^17 * 2^17 = 2^34 booleans = 16GB), so we do it in chunks
    # Instead, use the fact that words with weight < 6 are the "bad" differences
    bad_diffs = [d for d in range(MAX_WORD) if pc[d] < MIN_DIST]
    bad_diffs_arr = np.array(bad_diffs, dtype=np.uint32)

    for i in range(MAX_WORD):
        # j is in conflict with i iff j = i ^ d for some bad d
        # Build bitset: for each bad d, set bit (i^d)
        neighbors = i ^ bad_diffs_arr
        # Convert to bitset
        bs = 0
        for j in neighbors:
            bs |= (1 << j)
        conflict[i] = bs
    return conflict


def construct():
    start_time = time.time()

    # Build conflict matrix using a more efficient approach
    # Use numpy to compute popcounts, then for each word, find conflicts
    # Actually, let's use a smarter bit-parallel approach:
    # Represent the availability set as a Python int (bitset of MAX_WORD bits)

    # Precompute bad differences (weights < 6)
    bad_diffs = []
    for d in range(MAX_WORD):
        if bin(d).count('1') < MIN_DIST:
            bad_diffs.append(d)

    bad_diffs_np = np.array(bad_diffs, dtype=np.uint32)

    # Availability as a Python int bitset
    # bit i = 1 means word i is available
    all_available = (1 << MAX_WORD) - 1

    best_code = []
    best_size = 0
    max_restarts = 500

    for restart in range(max_restarts):
        if time.time() - start_time > 55:
            break

        avail = all_available
        code = []

        while True:
            if time.time() - start_time > 55.5:
                break

            # Find all available words
            # We need to find candidates and pick the one with fewest remaining conflicts
            # To do this efficiently, we need to know for each available word u,
            # how many other available words are in conflict with u.

            # First, extract available word indices
            # This is expensive for 2^17 words. Let's use a different strategy:
            # Sample a subset of available words for MRV heuristic

            # Get available indices - use a list comprehension which is slow but ok for 2^17
            # Better: maintain a list of available words and remove as we go
            # Let's use a set for O(1) removal

            # Actually, for performance, let's use a numpy boolean array
            break  # We'll restructure below

        # The above while loop is not efficient. Let's use a different approach.
        break

    # Revised approach: use numpy boolean array for availability
    # and periodically sample candidates for MRV

    best_code = []
    best_size = 0
    max_restarts = 2000

    for restart in range(max_restarts):
        if time.time() - start_time > 55:
            break

        is_avail = np.ones(MAX_WORD, dtype=bool)
        code = []

        while True:
            if time.time() - start_time > 55.5:
                break

            avail_idx = np.nonzero(is_avail)[0]
            if len(avail_idx) == 0:
                break

            # Sample for MRV
            sample_size = min(80, len(avail_idx))
            if sample_size == 0:
                break

            samples = np.random.choice(avail_idx, sample_size, replace=False)

            best_u = -1
            best_deg = float('inf')

            for u in samples:
                u_int = int(u)
                neighbors = u_int ^ bad_diffs_np
                deg = np.sum(is_avail[neighbors])
                if deg < best_deg:
                    best_deg = deg
                    best_u = u_int

            if best_u == -1:
                break

            code.append(best_u)

            # Mark u and its neighbors as unavailable
            neighbors = best_u ^ bad_diffs_np
            is_avail[neighbors] = False
            is_avail[best_u] = False

        if len(code) > best_size:
            best_size = len(code)
            best_code = code.copy()

    return best_code
