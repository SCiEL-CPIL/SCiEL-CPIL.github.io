from math import isqrt
import time

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill

    # Build smallest prime factor sieve for fast factorization
    limit = 10210
    spf = list(range(limit + 1))
    for i in range(2, int(limit**0.5) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Collect all primes up to limit
    primes = [i for i in range(2, limit + 1) if spf[i] == i]
    prime_index = {p: i for i, p in enumerate(primes)}
    num_primes = len(primes)

    # Target mask: 210 = 2 * 3 * 5 * 7, all to odd power
    target_mask = 0
    for p in [2, 3, 5, 7]:
        target_mask |= (1 << prime_index[p])

    # Precompute parity masks for all numbers 211..10210
    def get_mask(n):
        mask = 0
        temp = n
        while temp > 1:
            p = spf[temp]
            count = 0
            while temp % p == 0:
                temp //= p
                count += 1
            if count % 2 == 1:
                mask |= (1 << prime_index[p])
        return mask

    # Precompute masks for the range we'll search
    max_M = 300  # search up to M=300 if needed
    masks_cache = {}
    for x in range(211, max_M + 1):
        masks_cache[x] = get_mask(x)

    def verify(selected):
        """Verify that 210 * prod(selected) is a perfect square."""
        prod = 210
        for x in selected:
            prod *= x
        s = isqrt(prod)
        return s * s == prod

    # Try small subset sizes first for each M
    # Size 1: need a number whose mask equals target_mask
    # Size 2: need two numbers whose masks XOR to target_mask
    # Size 3: need three numbers whose masks XOR to target_mask
    
    for M in range(211, max_M + 1):
        if time.time() - start_time > deadline:
            break

        nums = list(range(211, M + 1))
        n = len(nums)
        masks = [masks_cache[x] for x in nums]

        # Check size 1
        for i in range(n):
            if masks[i] == target_mask:
                result = [nums[i]]
                if verify(result):
                    return result

        # Check size 2
        mask_to_idx = {}
        for i in range(n):
            needed = target_mask ^ masks[i]
            if needed in mask_to_idx:
                j = mask_to_idx[needed]
                if j != i:
                    result = [nums[j], nums[i]]
                    if verify(result):
                        return result
            if masks[i] not in mask_to_idx:
                mask_to_idx[masks[i]] = i

        # Check size 3 using meet-in-the-middle on pairs
        # For each pair (i,j) with i<j, compute mask_i ^ mask_j, store in dict
        # Then for each k, check if mask_k ^ target_mask is in the pair dict
        pair_mask_to_pair = {}
        for i in range(n):
            for j in range(i + 1, n):
                pm = masks[i] ^ masks[j]
                if pm not in pair_mask_to_pair:
                    pair_mask_to_pair[pm] = (i, j)

        for k in range(n):
            needed = target_mask ^ masks[k]
            if needed in pair_mask_to_pair:
                i, j = pair_mask_to_pair[needed]
                if i != k and j != k:
                    result = [nums[i], nums[j], nums[k]]
                    if verify(result):
                        return result

        # Meet-in-the-middle for larger subsets
        mid = n // 2
        left_masks = masks[:mid]
        right_masks = masks[mid:]

        # Generate all subset XORs for left half
        left_sums = {}
        for i in range(1 << len(left_masks)):
            s = 0
            for j in range(len(left_masks)):
                if i & (1 << j):
                    s ^= left_masks[j]
            if s not in left_sums:
                left_sums[s] = i  # store bitmask of which elements

        # Search right half
        found = None
        for i in range(1 << len(right_masks)):
            s = 0
            for j in range(len(right_masks)):
                if i & (1 << j):
                    s ^= right_masks[j]
            needed = target_mask ^ s
            if needed in left_sums:
                left_bits = left_sums[needed]
                # Combine: ensure non-empty
                if left_bits or i:
                    # Build the subset
                    subset = []
                    for j in range(len(left_masks)):
                        if left_bits & (1 << j):
                            subset.append(j)
                    for j in range(len(right_masks)):
                        if i & (1 << j):
                            subset.append(mid + j)
                    found = subset
                    break

        if found is not None:
            selected_nums = [nums[idx] for idx in found]
            if verify(selected_nums):
                return selected_nums

    # Fallback
    return [224, 240]
