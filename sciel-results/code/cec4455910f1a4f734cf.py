def construct():
    import math
    import bisect
    from collections import defaultdict

    MAX_N = 10_000_000

    # 1. Sieve for Smallest Prime Factor (SPF)
    spf = list(range(MAX_N + 1))
    for i in range(2, int(math.isqrt(MAX_N)) + 1):
        if spf[i] == i:
            for j in range(i * i, MAX_N + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # 2. Compute sigma for all n
    sigma = [0] * (MAX_N + 1)
    sigma[1] = 1
    for n in range(2, MAX_N + 1):
        orig_n = n
        s = 1
        while n > 1:
            p = spf[n]
            power_sum = 1
            power = 1
            while n % p == 0:
                n //= p
                power *= p
                power_sum += power
            s *= power_sum
        sigma[orig_n] = s

    # 3. Group numbers by sigma
    sigma_groups = defaultdict(list)
    for n in range(1, MAX_N + 1):
        sigma_groups[sigma[n]].append(n)

    # 4. Find best pair
    best_a = None
    best_b = None
    best_num = 0
    best_den = 1

    for sig_val, nums in sigma_groups.items():
        if len(nums) < 2:
            continue
        nums.sort()
        k = len(nums)

        for i in range(k):
            b = nums[i]
            target = 1.5 * b

            pos = bisect.bisect_left(nums, target)

            candidates = []
            if pos < k:
                candidates.append(nums[pos])
            if pos > 0:
                candidates.append(nums[pos - 1])

            for a in candidates:
                if a == b:
                    continue
                num = abs(2 * a - 3 * b)
                if num == 0:
                    continue
                den = 2 * b

                if best_a is None or num * best_den < best_num * den:
                    best_a = a
                    best_b = b
                    best_num = num
                    best_den = den

    if best_a is None:
        return [11, 6]

    return [int(best_a), int(best_b)]
