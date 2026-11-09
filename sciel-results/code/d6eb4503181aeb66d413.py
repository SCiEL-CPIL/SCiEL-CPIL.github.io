import numpy as np
import time

def construct():
    t_start = time.time()
    deadline = t_start + 85.0

    N = 10_000_000

    # Step 1: Linear sieve for smallest prime factor
    spf = np.zeros(N + 1, dtype=np.int32)
    primes = []
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            primes.append(i)
        for p in primes:
            if p > spf[i] or i * p > N:
                break
            spf[i * p] = p

    if time.time() > deadline - 30.0:
        return [1, 1]

    # Step 2: Precompute sigma using multiplicative property
    sigma_arr = np.ones(N + 1, dtype=np.int64)
    p_pow = np.ones(N + 1, dtype=np.int64)
    p_sum = np.ones(N + 1, dtype=np.int64)

    for n in range(2, N + 1):
        if time.time() > deadline - 25.0:
            break
        p = spf[n]
        m = n // p
        if m % p == 0:
            p_pow[n] = p_pow[m] * p
            p_sum[n] = p_sum[m] + p_pow[n]
            sigma_arr[n] = sigma_arr[m] * p_sum[n] // p_sum[m]
        else:
            p_pow[n] = p
            p_sum[n] = 1 + p
            sigma_arr[n] = sigma_arr[m] * (1 + p)

    if time.time() > deadline - 15.0:
        return [1, 1]

    # Step 3: Build sigma groups for fast lookup
    # For each sigma value, store sorted list of numbers with that sigma
    # This allows binary search for candidates near 1.5*b

    # First, count how many numbers have each sigma value
    max_sigma = int(sigma_arr.max())
    sigma_counts = np.zeros(max_sigma + 1, dtype=np.int32)
    for n in range(1, N + 1):
        sigma_counts[sigma_arr[n]] += 1

    # Compute prefix sums for offset
    sigma_offsets = np.zeros(max_sigma + 1, dtype=np.int32)
    for s in range(1, max_sigma + 1):
        sigma_offsets[s] = sigma_offsets[s - 1] + sigma_counts[s - 1]

    # Fill the groups
    sigma_groups = np.zeros(N, dtype=np.int32)
    fill_pos = sigma_offsets.copy()
    for n in range(1, N + 1):
        s = sigma_arr[n]
        pos = fill_pos[s]
        sigma_groups[pos] = n
        fill_pos[s] += 1

    if time.time() > deadline - 10.0:
        return [1, 1]

    # Build list of (start, end) for each sigma value
    # sigma_groups[start:end] contains all numbers with that sigma
    group_starts = sigma_offsets
    group_ends = sigma_offsets + sigma_counts

    # Step 4: Search for best pair (a, b) with sigma(a) == sigma(b)
    # Minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # For each b, target a ≈ 1.5*b
    # Use binary search in the sigma group to find candidates near 1.5*b

    best_num = 0
    best_den = 1
    best_a = 1
    best_b = 1

    # Iterate over b from large to small
    # For each b, find sigma(b), then search in that group for a near 1.5*b
    # The error is |2a-3b|/(2b), so we want small |2a-3b| and large b

    # Check b from N down to 1
    # For each b, compute target_a = round(1.5*b)
    # Use binary search to find the position in the sigma group closest to target_a
    # Then check a few candidates around that position

    W = 5  # check W candidates on each side

    for b in range(N, 0, -1):
        if time.time() > deadline - 3.0:
            break

        sb = int(sigma_arr[b])
        if sb == 1:
            continue  # sigma(1) = 1, only n=1 has this

        start = int(group_starts[sb])
        end = int(group_ends[sb])
        if end - start < 2:
            continue  # need at least 2 numbers in the group

        # Target a
        if b % 2 == 0:
            target = 3 * b // 2  # exact, |2a-3b|=0, invalid
            # We need a != target, so check target ± 1, ± 2, ...
            target_lo = target - 1
            target_hi = target + 1
        else:
            target_lo = (3 * b - 1) // 2
            target_hi = (3 * b + 1) // 2

        # Binary search in sigma_groups[start:end] for target_lo
        # numpy searchsorted
        arr = sigma_groups[start:end]
        idx_lo = np.searchsorted(arr, target_lo)
        idx_hi = np.searchsorted(arr, target_hi)

        # Check candidates around idx_lo and idx_hi
        candidates = set()
        for idx in (idx_lo, idx_hi):
            for d in range(-W, W + 1):
                j = idx + d
                if 0 <= j < len(arr):
                    a = int(arr[j])
                    if a != b and 1 <= a <= N:
                        candidates.add(a)

        for a in candidates:
            num = abs(2 * a - 3 * b)
            if num == 0:
                continue
            den = 2 * b
            if best_num == 0 or num * best_den < best_num * den:
                best_num = num
                best_den = den
                best_a = a
                best_b = b

    if best_a == 1 and best_b == 1:
        return [1, 1]

    return [int(best_a), int(best_b)]
