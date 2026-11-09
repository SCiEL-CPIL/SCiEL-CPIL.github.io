import time

def construct():
    # Direct construction of 16 integers in [2, 40] with distinct subset products.
    # Strategy: Use prime powers with sum-free exponent sets per prime.
    # Primes in [2, 40]: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
    
    result = []
    
    # For prime 2: use exponents {3, 4, 5} -> 8, 16, 32
    result.extend([2**3, 2**4, 2**5])  # 8, 16, 32
    
    # For prime 3: use exponents {2, 3} -> 9, 27
    result.extend([3**2, 3**3])  # 9, 27
    
    # For prime 5: use exponents {1, 2} -> 5, 25
    result.extend([5**1, 5**2])  # 5, 25
    
    # For primes 7, 11, 13, 17, 19, 23, 29, 31, 37: use the prime itself
    for p in [7, 11, 13, 17, 19, 23, 29, 31, 37]:
        result.append(p)
    
    # Ensure all are Python ints, in range [2, 40], distinct
    result = [int(x) for x in result]
    assert len(result) == 16
    assert all(2 <= x <= 40 for x in result)
    assert len(set(result)) == 16
    
    # Verify the 16-element set is valid
    if not _verify_subset_products_distinct(result):
        # Fallback to a known smaller set if verification fails (should not happen)
        return [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        
    # Try to find a 17-element set by adding one more number
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 50.0  # 50 seconds limit
    
    candidates = [x for x in range(2, 41) if x not in result]
    
    for cand in candidates:
        if time.time() > deadline:
            break
            
        new_set = result + [cand]
        if _verify_subset_products_distinct(new_set):
            return new_set
            
    return result

def _verify_subset_products_distinct(nums):
    products = set()
    n = len(nums)
    # Iterate over all subsets using bitmasks
    for mask in range(1 << n):
        prod = 1
        for i in range(n):
            if mask & (1 << i):
                prod *= nums[i]
        if prod in products:
            return False
        products.add(prod)
    return True
