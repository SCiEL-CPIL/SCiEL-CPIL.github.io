def construct():
    import math
    
    MAX_N = 10_000_000
    
    # 1. Sieve for Smallest Prime Factor (SPF)
    spf = list(range(MAX_N + 1))
    for i in range(2, int(math.isqrt(MAX_N)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i*i, MAX_N + 1, i):
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

    # 3. Reverse Search
    # We iterate b from MAX_N down to 1.
    # For each b, we check a values closest to 1.5 * b.
    # The error is |a/b - 1.5| = |2a - 3b| / (2b).
    # To minimize error, we want |2a - 3b| to be 0 (invalid) or 1, and b to be as large as possible.
    # If we find a pair with |2a - 3b| = 1, it is the best possible for that b.
    # Since we iterate b downwards, the first such pair found is the global best.
    
    for b in range(MAX_N, 0, -1):
        s_b = sigma[b]
        
        # Ideal a is 1.5 * b
        # 2a = 3b
        # We check integers a such that 2a is close to 3b.
        # Let 3b = 2k + r, where r is 0 or 1.
        # If r == 0, 3b is even, so 1.5b is integer. a = 1.5b.
        # Then 2a - 3b = 0. Error 0. Invalid.
        # Next best are a = 1.5b - 1 and a = 1.5b + 1.
        # If r == 1, 3b is odd. 1.5b = k + 0.5.
        # Closest integers are k and k+1.
        # 2k - 3b = -1. 2(k+1) - 3b = 1.
        # In both cases, the closest integers give |2a - 3b| = 1.
        
        # So we just need to check the two integers closest to 1.5 * b.
        
        three_b = 3 * b
        # a1 = floor(1.5 * b)
        # a2 = ceil(1.5 * b)
        
        # 1.5 * b = 3b / 2
        # floor(3b / 2)
        a1 = three_b // 2
        a2 = a1 + 1
        
        # Check a1
        if 1 <= a1 <= MAX_N:
            if a1 != b and sigma[a1] == s_b:
                # Check if error is 0 (2a1 - 3b == 0)
                if 2 * a1 - 3 * b != 0:
                    return [a1, b]
        
        # Check a2
        if 1 <= a2 <= MAX_N:
            if a2 != b and sigma[a2] == s_b:
                if 2 * a2 - 3 * b != 0:
                    return [a2, b]

    # Fallback if no pair with error 1 found (unlikely)
    # Search for any valid pair
    # This part is slow, but shouldn't be reached
    from collections import defaultdict
    sigma_groups = defaultdict(list)
    for n in range(1, MAX_N + 1):
        sigma_groups[sigma[n]].append(n)
        
    best_error = float('inf')
    best_a, best_b = None, None
    
    for sig_val, nums in sigma_groups.items():
        if len(nums) < 2:
            continue
        nums.sort()
        k = len(nums)
        for i in range(k):
            b = nums[i]
            target = 1.5 * b
            import bisect
            pos = bisect.bisect_left(nums, target)
            candidates = []
            if pos < k:
                candidates.append(nums[pos])
            if pos > 0:
                candidates.append(nums[pos-1])
            for a in candidates:
                if a == b:
                    continue
                num = abs(2 * a - 3 * b)
                if num == 0:
                    continue
                den = 2 * b
                err = num / den
                if err < best_error:
                    best_error = err
                    best_a = a
                    best_b = b
                    
    if best_a is None:
        return [11, 6]
        
    return [best_a, best_b]
