import time

def construct():
    """
    Returns the maximum cardinality subset of [1, 255] containing 255
    such that gcd(a, b) > 1 for all distinct pairs a, b in the subset.
    
    The optimal strategy is to take all multiples of the smallest prime factor
    of 255, which is 3. This yields a set of size 85.
    Any attempt to include numbers not divisible by 3 requires excluding
    numbers that are "pure" multiples of 3 (not divisible by 5 or 17),
    which reduces the total size significantly (max 51 if switching to 5s).
    """
    # Deadline guard, though this operation is O(1) effectively
    start_time = time.time()
    
    # All multiples of 3 in [1, 255]
    # 3, 6, 9, ..., 255
    result = list(range(3, 256, 3))
    
    # Safety check: must contain 255
    if 255 not in result:
        # This should never happen for this specific range
        result.append(255)
        
    return result
