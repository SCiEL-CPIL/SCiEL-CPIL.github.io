import math
import time
from collections import defaultdict

def construct():
    deadline = time.time() + 85
    
    max_val = 10210
    min_val = 211
    
    # Sieve of Eratosthenes to get primes
    is_prime = [True] * (max_val + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_val**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, max_val + 1, i):
                is_prime[j] = False
    primes = [i for i, prime in enumerate(is_prime) if prime]
    
    # Map prime to index for bitmask
    prime_to_idx = {p: i for i, p in enumerate(primes)}
    
    def get_mask(n):
        mask = 0
        d = 2
        temp = n
        while d * d <= temp:
            if temp % d == 0:
                exp = 0
                while temp % d == 0:
                    temp //= d
                    exp += 1
                if exp % 2 == 1:
                    if d in prime_to_idx:
                        mask |= (1 << prime_to_idx[d])
            d += 1 if d == 2 else 2
        if temp > 1:
            if temp in prime_to_idx:
                mask |= (1 << prime_to_idx[temp])
        return mask

    # Precompute masks for all n in [min_val, max_val]
    masks = [0] * (max_val + 1)
    for n in range(min_val, max_val + 1):
        masks[n] = get_mask(n)
        
    target_mask = get_mask(210)
    
    # Build map from mask to sorted list of numbers
    mask_to_nums = defaultdict(list)
    for n in range(min_val, max_val + 1):
        mask_to_nums[masks[n]].append(n)
        
    best_max = max_val + 1
    best_set = None
    
    def try_set(s):
        nonlocal best_max, best_set
        m = max(s)
        if m < best_max:
            best_max = m
            best_set = list(s)
    
    # Size 1
    if target_mask in mask_to_nums:
        nums = mask_to_nums[target_mask]
        if nums:
            try_set([nums[0]])
                
    # Size 2: check multiple pairs to find min max
    seen_mask_pairs = set()
    for m in mask_to_nums.keys():
        comp_m = m ^ target_mask
        if comp_m in mask_to_nums:
            pair_key = (min(m, comp_m), max(m, comp_m))
            if pair_key in seen_mask_pairs:
                continue
            seen_mask_pairs.add(pair_key)
            
            nums1 = mask_to_nums[m]
            nums2 = mask_to_nums[comp_m]
            
            if not nums1 or not nums2:
                continue
                
            if m == comp_m:
                if len(nums1) >= 2:
                    for i in range(min(5, len(nums1))):
                        for j in range(i+1, min(5, len(nums1))):
                            try_set([nums1[i], nums1[j]])
            else:
                for n1 in nums1[:5]:
                    for n2 in nums2[:5]:
                        if n1 != n2:
                            try_set([n1, n2])
                
    # Size 3: mask-based search
    mask_list = sorted(mask_to_nums.keys(), key=lambda m: min(mask_to_nums[m]))
    for i in range(len(mask_list)):
        if time.time() > deadline - 15:
            break
        m1 = mask_list[i]
        nums1 = mask_to_nums[m1]
        if not nums1:
            continue
        for j in range(i + 1, len(mask_list)):
            if time.time() > deadline - 15:
                break
            m2 = mask_list[j]
            nums2 = mask_to_nums[m2]
            if not nums2:
                continue
            # Early termination: if min possible max is already >= best_max, skip
            min_m1 = nums1[0]
            min_m2 = nums2[0]
            if min_m1 >= best_max or min_m2 >= best_max:
                continue
            m3 = target_mask ^ m1 ^ m2
            if m3 in mask_to_nums:
                nums3 = mask_to_nums[m3]
                if not nums3:
                    continue
                min_m3 = nums3[0]
                if min_m3 >= best_max:
                    continue
                # Find best combination
                for n1 in nums1[:3]:
                    if n1 >= best_max:
                        break
                    for n2 in nums2[:3]:
                        if n2 >= best_max or n2 == n1:
                            continue
                        for n3 in nums3[:3]:
                            if n3 >= best_max or n3 == n1 or n3 == n2:
                                continue
                            try_set([n1, n2, n3])
                            break
                        break
                    break
    
    # Size 4: mask-based search
    for i in range(len(mask_list)):
        if time.time() > deadline - 5:
            break
        m1 = mask_list[i]
        nums1 = mask_to_nums[m1]
        if not nums1 or nums1[0] >= best_max:
            continue
        for j in range(i + 1, len(mask_list)):
            if time.time() > deadline - 5:
                break
            m2 = mask_list[j]
            nums2 = mask_to_nums[m2]
            if not nums2 or nums2[0] >= best_max:
                continue
            for k in range(j + 1, len(mask_list)):
                if time.time() > deadline - 5:
                    break
                m3 = mask_list[k]
                nums3 = mask_to_nums[m3]
                if not nums3 or nums3[0] >= best_max:
                    continue
                m4 = target_mask ^ m1 ^ m2 ^ m3
                if m4 in mask_to_nums:
                    nums4 = mask_to_nums[m4]
                    if not nums4 or nums4[0] >= best_max:
                        continue
                    for n1 in nums1[:2]:
                        if n1 >= best_max:
                            break
                        for n2 in nums2[:2]:
                            if n2 >= best_max or n2 == n1:
                                continue
                            for n3 in nums3[:2]:
                                if n3 >= best_max or n3 == n1 or n3 == n2:
                                    continue
                                for n4 in nums4[:2]:
                                    if n4 >= best_max or n4 in (n1, n2, n3):
                                        continue
                                    try_set([n1, n2, n3, n4])
                                    break
                                break
                            break
                        break
    
    # Polish: swap elements with same-mask smaller elements to reduce max
    if best_set is not None:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for idx in range(len(best_set)):
                old_val = best_set[idx]
                old_mask = masks[old_val]
                if old_mask in mask_to_nums:
                    candidates = mask_to_nums[old_mask]
                    for cand in candidates:
                        if cand < old_val and cand not in best_set:
                            best_set[idx] = cand
                            improved = True
                            break
    
    if best_set is None:
        best_set = [840]
        best_max = 840
        
    # Verify
    prod = 210
    for x in best_set:
        prod *= x
        
    x_sqrt = math.isqrt(prod)
    if x_sqrt * x_sqrt != prod:
        best_set = [840]
        best_max = 840
        
    return [int(x) for x in best_set]
