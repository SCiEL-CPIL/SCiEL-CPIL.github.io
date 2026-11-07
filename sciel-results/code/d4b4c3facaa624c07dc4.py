import time
from collections import defaultdict
import bisect
import numpy as np
from fractions import Fraction

def construct():
    N = 10000000
    start = time.time()
    deadline = start + 85

    # Compute sigma(n) for all n up to N using a sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sigma[d::d] += d
        if time.time() > deadline - 15:
            break

    # Convert to list for grouping
    sigma_list = sigma.tolist()
    
    # Group by sigma value
    groups = defaultdict(list)
    for n in range(1, N + 1):
        if time.time() > deadline - 10:
            break
        groups[sigma_list[n]].append(n)
    
    # Find best pair
    best_err = float('inf')
    best_pair = (11, 6)  # fallback from seed
    
    # For each sigma group, find pairs where a/b ≈ 3/2
    # i.e., a ≈ 1.5 * b, so for each b, look for a near 1.5*b
    for s, nums in groups.items():
        if time.time() > deadline - 5:
            break
        if len(nums) < 2:
            continue
        nums.sort()
        for b in nums:
            if time.time() > deadline - 3:
                break
            target = 3 * b // 2  # a should be close to 3b/2
            # Search for a in nums near target
            idx = bisect.bisect_left(nums, target)
            candidates = []
            # Check a few neighbors around the target
            for offset in range(-5, 6):
                ci = idx + offset
                if 0 <= ci < len(nums):
                    candidates.append(nums[ci])
            for a in candidates:
                if a == b:
                    continue
                # Exact error: |a/b - 3/2| = |2a - 3b| / (2b)
                err_num = abs(2 * a - 3 * b)
                err_den = 2 * b
                err = err_num / err_den
                if err < best_err:
                    best_err = err
                    best_pair = (a, b)
                    if best_err == 0:
                        return [int(a), int(b)]
    
    # If we still have time, do a more exhaustive search on the best sigma groups
    if time.time() < deadline - 3:
        # Sort groups by size and focus on larger ones
        sorted_groups = sorted(groups.items(), key=lambda x: -len(x[1]))
        for s, nums in sorted_groups[:200]:
            if time.time() > deadline - 2:
                break
            if len(nums) < 2:
                continue
            nums.sort()
            n = len(nums)
            # For each b, find closest a to 3b/2
            for i, b in enumerate(nums):
                if time.time() > deadline - 1:
                    break
                target = 3 * b / 2
                idx = bisect.bisect_left(nums, target)
                # Check neighbors
                for offset in range(-10, 11):
                    ci = idx + offset
                    if 0 <= ci < n:
                        a = nums[ci]
                        if a == b:
                            continue
                        err_num = abs(2 * a - 3 * b)
                        err_den = 2 * b
                        err = err_num / err_den
                        if err < best_err:
                            best_err = err
                            best_pair = (a, b)
                            if best_err == 0:
                                return [int(a), int(b)]
    
    return [int(best_pair[0]), int(best_pair[1])]
