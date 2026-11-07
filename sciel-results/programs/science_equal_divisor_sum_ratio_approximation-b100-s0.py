import numpy as np
import time
from bisect import bisect_left

P = {'maximum': 10000000, 'target': [3, 2]}

def construct():
    start_time = time.time()
    deadline = start_time + 75.0

    N = 10000000

    sigma = np.zeros(N + 1, dtype=np.int64)

    half = N // 2
    for i in range(1, half + 1):
        if time.time() > deadline - 25.0:
            return [11, 6]
        sigma[i::i] += i

    sigma[half+1:] += np.arange(half+1, N+1, dtype=np.int64)

    if time.time() > deadline - 15.0:
        return [11, 6]

    groups = {}
    for n in range(1, N + 1):
        s = int(sigma[n])
        if s not in groups:
            groups[s] = []
        groups[s].append(n)

    best_value = 1.0
    best_pair = (11, 6)

    for s, nums in groups.items():
        if len(nums) < 2:
            continue
        if time.time() > deadline - 3.0:
            break
        nums.sort()
        nlen = len(nums)
        for b_idx in range(nlen):
            b = nums[b_idx]
            if b_idx % 500 == 0 and time.time() > deadline - 3.0:
                break
            target = int(1.5 * b)
            idx = bisect_left(nums, target)
            candidates = set()
            if idx < nlen:
                candidates.add(nums[idx])
            if idx > 0:
                candidates.add(nums[idx-1])
            for a in candidates:
                if a == b:
                    continue
                num = abs(2*a - 3*b)
                if num == 0:
                    continue
                val = num / (2.0 * b)
                if val < best_value:
                    best_value = val
                    best_pair = (a, b)

    return [int(best_pair[0]), int(best_pair[1])]
# EVOLVE-BLOCK-END
