import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    MAX_N = 10000000
    
    # Use bytearray for is_comp to save memory and improve cache performance
    is_comp = bytearray(MAX_N + 1)
    sigma = [0] * (MAX_N + 1)
    spow = [0] * (MAX_N + 1)
    sval = [0] * (MAX_N + 1)
    primes = []
    
    sigma[1] = 1
    spow[1] = 1
    sval[1] = 1
    
    # Linear Sieve for sigma
    for i in range(2, MAX_N + 1):
        if not is_comp[i]:
            primes.append(i)
            spow[i] = i
            sval[i] = 1 + i
            sigma[i] = 1 + i
        
        for p in primes:
            j = i * p
            if j > MAX_N:
                break
            is_comp[j] = 1
            if i % p == 0:
                spow[j] = spow[i] * p
                sval[j] = sval[i] + spow[j]
                sigma[j] = (sigma[i] // sval[i]) * sval[j]
                break
            else:
                spow[j] = p
                sval[j] = 1 + p
                sigma[j] = sigma[i] * (1 + p)
                
    if time.time() > deadline - 15:
        # Fallback if sieve takes too long (unlikely for 10^7)
        return [11, 6]

    # Group by sigma
    fibers = defaultdict(list)
    for n in range(1, MAX_N + 1):
        fibers[sigma[n]].append(n)
        
    if time.time() > deadline - 10:
        return [11, 6]

    best_a, best_b = 11, 6
    # Exact error for 11/6: |11/6 - 3/2| = |11/6 - 9/6| = 2/6 = 1/3
    best_err_num = 1
    best_err_den = 3
    
    # Process fibers
    # Sort fibers by size descending to prioritize larger fibers which may contain better ratios
    # Or sort by max value? 
    # Let's sort by size descending. Larger fibers have more chances of having a pair close to 3/2.
    fiber_list = []
    for s_val, nums in fibers.items():
        if len(nums) < 2:
            continue
        fiber_list.append((s_val, nums))
        
    fiber_list.sort(key=lambda x: len(x[1]), reverse=True)
    
    if time.time() > deadline - 5:
        return [int(best_a), int(best_b)]

    for s_val, nums in fiber_list:
        if time.time() > deadline - 1:
            break
            
        # Sort nums for two-pointer search
        nums.sort()
        n = len(nums)
        
        # Two-pointer search for min |a/b - 3/2| => min |2a - 3b|
        # We want a/b ≈ 3/2 => 2a ≈ 3b => a ≈ 1.5b
        # Let i point to a, j point to b.
        # Since a and b are from the same list, we can use two pointers i, j.
        # If 2*nums[i] < 3*nums[j], we need a larger a, so increment i.
        # If 2*nums[i] > 3*nums[j], we need a smaller a (or larger b? no, 3b is larger), so increment j.
        # Wait, let's trace:
        # We want to minimize |2a - 3b|.
        # Start i=0, j=0.
        # If 2*nums[i] < 3*nums[j], then 2a < 3b => a/b < 1.5. To increase a/b, increase a (i++).
        # If 2*nums[i] > 3*nums[j], then 2a > 3b => a/b > 1.5. To decrease a/b, decrease a (i--) or increase b (j++).
        # Since we want to explore the space, standard two pointer for closest pair in sorted arrays:
        # i starts at 0, j starts at n-1? No, that's for sum.
        # Here we want a/b close to 1.5.
        # Let's iterate j from 0 to n-1 (b), and maintain i such that nums[i] is closest to 1.5 * nums[j].
        # Since nums is sorted, as j increases, 1.5 * nums[j] increases. So i is non-decreasing.
        # This is a standard "merge" style two-pointer scan.
        
        i = 0
        for j in range(n):
            # Advance i while nums[i] < 1.5 * nums[j]
            # Actually, we want to check the closest one.
            # Let's just find the position where 2*nums[k] is closest to 3*nums[j].
            # Since 2*nums[k] is increasing with k, we can advance i.
            
            while i < n and 2 * nums[i] < 3 * nums[j]:
                i += 1
                
            # Check i and i-1
            if i < n and i != j:
                a = nums[i]
                b = nums[j]
                num_diff = abs(2 * a - 3 * b)
                # Error is num_diff / (2 * b)
                # Compare num_diff / (2 * b) < best_err_num / best_err_den
                # num_diff * best_err_den < best_err_num * 2 * b
                if num_diff * best_err_den < best_err_num * 2 * b:
                    best_err_num = num_diff
                    best_err_den = 2 * b
                    best_a = a
                    best_b = b
                    if best_err_num == 0:
                        return [int(best_a), int(best_b)]
                        
            if i > 0 and (i - 1) != j:
                a = nums[i - 1]
                b = nums[j]
                num_diff = abs(2 * a - 3 * b)
                if num_diff * best_err_den < best_err_num * 2 * b:
                    best_err_num = num_diff
                    best_err_den = 2 * b
                    best_a = a
                    best_b = b
                    if best_err_num == 0:
                        return [int(best_a), int(best_b)]
                        
    return [int(best_a), int(best_b)]
