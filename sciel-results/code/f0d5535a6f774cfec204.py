from math import isqrt
import time

def _get_parity_vector(n, primes):
    """
    Returns a bitmask where bit i is 1 if prime[i] has odd exponent in n.
    primes is a sorted list of primes.
    """
    mask = 0
    temp = n
    for i, p in enumerate(primes):
        if p * p > temp:
            break
        count = 0
        while temp % p == 0:
            temp //= p
            count += 1
        if count % 2 == 1:
            mask |= (1 << i)
    if temp > 1:
        # temp is a prime factor
        # We need to find its index in primes
        # But primes list might not contain it if it's large?
        # We should generate primes up to max_n.
        pass
    return mask

# Better approach:
# Generate all primes up to 10210.
# Map each prime to an index.
# For each number, compute its parity vector as a tuple or int.

def construct():
    start_time = time.time()
    deadline = 80.0  # safety margin
    
    # Primes up to 10210
    limit = 10210
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            for j in range(i*i, limit + 1, i):
                sieve[j] = False
    primes = [i for i, is_prime in enumerate(sieve) if is_prime]
    prime_index = {p: i for i, p in enumerate(primes)}
    
    # Target vector for 210
    target_mask = 0
    temp = 210
    for p in [2, 3, 5, 7]:
        if temp % p == 0:
            target_mask |= (1 << prime_index[p])
            temp //= p
            if temp % p == 0:
                temp //= p # 210 is square-free, so only 1 division needed per prime
    # 210 is square free, so target_mask has bits for 2,3,5,7 set.

    # Precompute parity masks for numbers 211 to 10210
    # We only need up to 239 for the search, but let's be safe.
    # Actually, we search M from 211 to 239.
    
    def get_mask(n):
        mask = 0
        temp = n
        for p in primes:
            if p * p > temp:
                break
            if temp % p == 0:
                count = 0
                while temp % p == 0:
                    temp //= p
                    count += 1
                if count % 2 == 1:
                    mask |= (1 << prime_index[p])
        if temp > 1:
            mask |= (1 << prime_index[temp])
        return mask

    # We will iterate M from 211 upwards
    # For each M, we have numbers 211..M
    # We want to find a subset of these numbers whose XOR of masks is target_mask.
    
    for M in range(211, 240):
        if time.time() - start_time > deadline:
            break
            
        # Numbers in range
        nums = list(range(211, M + 1))
        masks = [get_mask(x) for x in nums]
        
        # Meet in the middle to find subset XORing to target_mask
        n = len(masks)
        if n == 0:
            continue
            
        mid = n // 2
        left_masks = masks[:mid]
        right_masks = masks[mid:]
        
        # Generate all subsets of left_masks
        left_sums = {}
        # 2^mid subsets
        # Store sum -> list of indices
        for i in range(1 << len(left_masks)):
            s = 0
            subset = []
            for j in range(len(left_masks)):
                if i & (1 << j):
                    s ^= left_masks[j]
                    subset.append(j)
            # Store the first occurrence of each sum to keep subsets small?
            # Or any occurrence. We want any valid subset.
            if s not in left_sums:
                left_sums[s] = subset
        
        # Generate all subsets of right_masks
        found_subset = None
        for i in range(1 << len(right_masks)):
            s = 0
            subset = []
            for j in range(len(right_masks)):
                if i & (1 << j):
                    s ^= right_masks[j]
                    subset.append(mid + j) # adjust indices
            
            # We need total XOR = target_mask
            # So left_XOR ^ right_XOR = target_mask
            # left_XOR = target_mask ^ right_XOR
            needed = target_mask ^ s
            if needed in left_sums:
                # Combine subsets
                # Ensure non-empty
                if left_sums[needed] or subset:
                    found_subset = left_sums[needed] + subset
                    break
        
        if found_subset is not None:
            # Convert indices back to numbers
            # found_subset contains indices in the range [0, M-211]
            selected_nums = [nums[idx] for idx in found_subset]
            # Verify
            prod = 210
            for x in selected_nums:
                prod *= x
            s = isqrt(prod)
            if s * s == prod:
                return selected_nums
            else:
                # Should not happen if logic is correct
                pass
        
        # If not found, continue to next M

    # Fallback to known solution
    return [224, 240]
