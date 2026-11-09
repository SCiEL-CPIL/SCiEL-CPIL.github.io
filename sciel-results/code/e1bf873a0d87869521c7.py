import time
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = 82.0

    MAX_N = 10_000_000

    # Sieve for sigma(n): sum of all positive divisors
    sigma = [0] * (MAX_N + 1)
    for i in range(1, MAX_N + 1):
        for j in range(i, MAX_N + 1, i):
            sigma[j] += i

    # Group numbers by their sigma value
    sigma_groups = defaultdict(list)
    for n in range(1, MAX_N + 1):
        sigma_groups[sigma[n]].append(n)

    best_error = float('inf')
    best_pair = (11, 6)

    # Only consider groups with at least 2 members
    groups = [(s, nums) for s, nums in sigma_groups.items() if len(nums) >= 2]
    # Process larger groups first
    groups.sort(key=lambda x: -len(x[1]))

    for s, nums in groups:
        if time.time() - start_time > DEADLINE:
            break
        nums.sort()
        n_len = len(nums)

        for b in nums:
            ideal_a = (3 * b) // 2
            # Binary search for ideal_a
            lo, hi = 0, n_len - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if nums[mid] < ideal_a:
                    lo = mid + 1
                else:
                    hi = mid
            # Check lo and lo-1
            if lo < n_len:
                a = nums[lo]
                if a != b:
                    err_num = abs(2 * a - 3 * b)
                    if err_num > 0:
                        err = err_num / (2 * b)
                        if err < best_error:
                            best_error = err
                            best_pair = (a, b)
            if lo - 1 >= 0:
                a = nums[lo - 1]
                if a != b:
                    err_num = abs(2 * a - 3 * b)
                    if err_num > 0:
                        err = err_num / (2 * b)
                        if err < best_error:
                            best_error = err
                            best_pair = (a, b)

        if time.time() - start_time > DEADLINE:
            break

    a, b = best_pair
    return [int(a), int(b)]
