import numpy as np
import time

def construct():
    t_start = time.time()
    deadline = t_start + 88.0

    N = 10_000_000

    # Step 1: Sieve for smallest prime factor (compact)
    spf = np.zeros(N + 1, dtype=np.int32)
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            if i * i <= N:
                for j in range(i * i, N + 1, i):
                    if spf[j] == 0:
                        spf[j] = i

    if time.time() > deadline - 30.0:
        return [1, 1]

    # Step 2: Precompute sigma using multiplicative property (needed for fast lookup)
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

    if time.time() > deadline - 20.0:
        return [1, 1]

    # Step 3: Build a dictionary mapping sigma -> list of numbers with that sigma
    # Only keep groups with at least 2 members to save memory
    sigma_groups = {}
    for n in range(1, N + 1):
        s = int(sigma_arr[n])
        if s in sigma_groups:
            sigma_groups[s].append(n)
        else:
            sigma_groups[s] = [n]

    if time.time() > deadline - 15.0:
        return [1, 1]

    # Step 4: Search for best pair (a, b) with sigma(a) == sigma(b)
    # Minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # For each b, target a ≈ 1.5*b
    # Check candidates near 1.5*b in the same sigma group

    best_num = 0  # numerator of |2a-3b|
    best_den = 1  # denominator = 2*b
    best_a = 1
    best_b = 1

    # Sort groups by size descending - larger groups are more likely to have good pairs
    # But we need to be efficient. Let's iterate over b and check if 1.5b is in same group.
    
    # Strategy: For each b from small to large, compute target_a = 3*b//2 or nearby
    # Check if any a in the same sigma group as b is close to 1.5*b
    
    # To make this efficient, for each sigma group, we sort the numbers and
    # for each b in the group, binary search for a ≈ 1.5*b
    
    for s, nums in sigma_groups.items():
        if time.time() > deadline - 2.0:
            break
        if len(nums) < 2:
            continue
        nums.sort()
        # For each b in nums, find a in nums closest to 1.5*b
        for b in nums:
            if time.time() > deadline - 1.0:
                break
            target = 1.5 * b
            # Binary search in nums
            lo, hi = 0, len(nums) - 1
            best_local_num = None
            best_local_den = None
            best_local_a = None
            # Check a few candidates around the target
            # Use bisect-like approach
            import bisect
            pos = bisect.bisect_left(nums, int(target))
            candidates = set()
            for offset in range(-3, 4):
                idx = pos + offset
                if 0 <= idx < len(nums):
                    candidates.add(nums[idx])
            for a in candidates:
                if a == b:
                    continue
                num = abs(2 * a - 3 * b)
                if num == 0:
                    continue  # exact 3/2 is invalid
                den = 2 * b
                # Compare num/den with best_num/best_den using cross multiplication
                if best_local_num is None:
                    best_local_num = num
                    best_local_den = den
                    best_local_a = a
                else:
                    # num/den < best_local_num/best_local_den  <=>  num*best_local_den < best_local_num*den
                    if num * best_local_den < best_local_num * den:
                        best_local_num = num
                        best_local_den = den
                        best_local_a = a
            if best_local_num is not None:
                # Compare with global best
                if best_num == 0:
                    best_num = best_local_num
                    best_den = best_local_den
                    best_a = best_local_a
                    best_b = b
                else:
                    if best_local_num * best_den < best_num * best_local_den:
                        best_num = best_local_num
                        best_den = best_local_den
                        best_a = best_local_a
                        best_b = b

    if best_a == 1 and best_b == 1:
        return [1, 1]

    return [int(best_a), int(best_b)]
