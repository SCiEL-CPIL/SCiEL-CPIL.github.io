"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import numpy as np

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23]

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # safety margin vs 90s kill

    lower = 1000
    upper = 1000000
    window_size = 24
    max_val = upper + window_size

    nvals = max_val - lower + 1

    # ---- Step 1: NumPy sieve for the 23-smooth part of each integer ----
    sp_part = np.ones(nvals, dtype=np.int64)
    for p in _SMALL_PRIMES:
        power = p
        while power <= max_val:
            first_mult_power = ((lower + power - 1) // power) * power
            if first_mult_power > max_val:
                break
            idx_array_power = np.arange(first_mult_power, max_val + 1, power) - lower
            sp_part[idx_array_power] *= p
            power *= p
            if power > max_val:
                break

    if time.time() > deadline:
        return lower

    # ---- Step 2: Map smooth-part values to compact integer IDs ----
    # Few distinct smooth-parts exist; assign a small int id to each unique value.
    uniq_vals = np.unique(sp_part)
    val_to_id = {int(v): i for i, v in enumerate(uniq_vals)}
    ids = np.array([val_to_id[int(v)] for v in sp_part], dtype=np.int32)

    if time.time() > deadline:
        return lower

    # ---- Step 3: Vectorized exact sliding-window distinct count ----
    # For a start position i (0-indexed into ids), the window covers ids[i : i+24]
    # which corresponds to integers (lower+i+1) .. (lower+i+24) for n = lower+i.
    # We need distinct count of the 24 ids in the window.
    #
    # Exact vectorized method per block of B start positions:
    #   M[j, t] = ids[s + t + j]  for j in 0..23, t in 0..B-1   (shape 24 x B)
    #   A value at row j is "new in window" iff it differs from every row k < j
    #   in the same column. distinct[t] = sum_j new[j, t].
    #
    # This is O(24^2 * B) per block, vectorized; total O(24^2 * N), fast.

    N = nvals  # number of ids; valid start positions i in [0, N - 24]
    max_start = N - window_size  # inclusive upper bound on start index
    # start i corresponds to n = lower + i, must have n <= upper
    # n = lower + i <= upper  =>  i <= upper - lower = 999000
    # also i <= max_start. Take the smaller.
    max_start = min(max_start, upper - lower)

    best_n = lower
    best_count = window_size + 1  # worst case all distinct (24) -> init to 25

    # Evaluate the very first window (i=0) to initialize best.
    first_window = ids[0:window_size]
    c0 = len(np.unique(first_window))
    best_count = c0
    best_n = lower

    BLOCK = 200000
    t = 0
    while t <= max_start:
        if time.time() > deadline:
            break
        b = min(BLOCK, max_start - t + 1)
        # Build M: shape (24, b)
        # M[j, k] = ids[t + k + j]
        # Use a strided view if possible; simplest: gather.
        base = t
        # indices: for row j, columns k -> ids[base + k + j]
        # Build via broadcasting: col_idx = np.arange(b); M[j] = ids[base + col_idx + j]
        col_idx = np.arange(b, dtype=np.int64)
        # M as list of rows then stack (24 x b)
        M = np.empty((window_size, b), dtype=np.int32)
        for j in range(window_size):
            M[j] = ids[base + col_idx + j]

        # new[j] : bool array length b, True if M[j] differs from all M[k] for k<j
        distinct = np.zeros(b, dtype=np.int32)
        # j = 0 always new
        distinct += 1
        for j in range(1, window_size):
            # compare M[j] against M[0..j-1]
            # eq shape (j, b)
            eq = (M[0:j] == M[j][None, :])
            any_eq = np.any(eq, axis=0)
            distinct += (~any_eq).astype(np.int32)

        # distinct[k] corresponds to start index i = t + k, n = lower + t + k
        # Find min in this block
        # We need the global best; track within block.
        # Use argmin but also need to update best_count/best_n.
        block_min = int(distinct.min())
        if block_min < best_count:
            # find first occurrence of block_min
            idx_local = int(np.argmin(distinct))
            i = t + idx_local
            n = lower + i
            best_count = block_min
            best_n = n
            # If we hit the theoretical floor, we could stop; 1 is impossible.
            # 2 is possible? keep scanning for safety but it's very good.
            if best_count == 2:
                # 1 is impossible for 24 consecutive ints (parity + odd parts differ),
                # so 2 is the global minimum. Stop.
                break

        t += b

    return best_n
# EVOLVE-BLOCK-END
