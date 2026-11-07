import time
import math
from collections import defaultdict

# EVOLVE-BLOCK-START
P = {'maximum': 10000000, 'target': [3, 2]}

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 seconds limit for safety
    
    N = P['maximum']
    sigma = [0] * (N + 1)
    
    # Sieve for sigma(n)
    # sigma(n) = sum of divisors of n
    # We can compute this by iterating d from 1 to N and adding d to sigma[m] for all m multiple of d.
    # This is O(N log N).
    for d in range(1, N + 1):
        for m in range(d, N + 1, d):
            sigma[m] += d
        
        # Check time periodically
        if d % 1000000 == 0:
            if time.time() > deadline:
                break

    # If we broke out early, we might have incomplete sigma. 
    # However, for the first few million, it should be fine. 
    # Actually, let's just assume we finish if N=10^7. 
    # O(10^7 log 10^7) is roughly 10^7 * 16 = 16 * 10^7 operations. 
    # In Python, this loop might be too slow. 
    # Let's optimize the sieve.
    
    # A faster way: 
    # Initialize sigma with 0.
    # for d in 1..N: for m in d..N step d: sigma[m] += d
    # This is standard. Python loops are slow. 
    # Can we use NumPy? 
    # sigma = np.zeros(N+1)
    # for d in range(1, N+1):
    #     sigma[d::d] += d
    # This is vectorized and much faster.
    
    # Let's restart with NumPy if available, otherwise stick to pure Python but maybe smaller N?
    # The task says N <= 10^7.
    # Let's try to implement the NumPy version.
    
    try:
        import numpy as np
        sigma = np.zeros(N + 1, dtype=np.int64)
        for d in range(1, N + 1):
            sigma[d::d] += d
            if d % 1000000 == 0 and time.time() > deadline:
                break
    except ImportError:
        # Fallback to pure python, but this will likely time out for N=10^7
        # We'll just use a smaller N if NumPy fails? No, N is fixed.
        # Pure python sieve for 10^7 is ~10-20 seconds if optimized? 
        # Actually, the nested loop in pure Python is very slow.
        # Let's hope NumPy is available as per task description.
        pass

    if time.time() > deadline:
        # Return a known good pair if we ran out of time during precomputation
        # [11, 6] gives sigma(11)=12, sigma(6)=12. |11/6 - 1.5| = |1.833 - 1.5| = 0.333
        # Is there a better one?
        # Let's return [11, 6] as fallback.
        return [11, 6]

    # Convert to list for faster access in the grouping phase? 
    # Or keep as numpy array.
    # We need to group indices by sigma value.
    # Creating a dictionary with keys as sigma values (up to ~10^8) and values as lists of indices.
    
    # This memory might be an issue. 
    # Max sigma for n=10^7 is roughly n log log n ~ 10^7 * 3 = 3*10^7.
    # Number of unique sigma values is less than N.
    
    groups = defaultdict(list)
    for i in range(1, N + 1):
        s = int(sigma[i])
        groups[s].append(i)
        
    if time.time() > deadline:
        return [11, 6]

    best_err = float('inf')
    best_pair = [11, 6]
    
    target_ratio = 1.5
    
    # Iterate through groups
    for s, nums in groups.items():
        if len(nums) < 2:
            continue
            
        # Sort nums
        nums.sort()
        
        # For each b in nums, find a in nums such that a/b is closest to 1.5
        # a is closest to 1.5 * b
        # Since nums is sorted, we can binary search for 1.5 * b
        
        for b in nums:
            target_a = 1.5 * b
            
            # Binary search in nums for target_a
            # We want a in nums closest to target_a
            # bisect_left gives the first element >= target_a
            # We check that element and the one before it.
            
            lo = 0
            hi = len(nums)
            
            # Standard binary search for lower bound
            while lo < hi:
                mid = (lo + hi) // 2
                if nums[mid] < target_a:
                    lo = mid + 1
                else:
                    hi = mid
            
            # lo is the index of the first element >= target_a
            # Check lo and lo-1
            
            candidates = []
            if lo < len(nums):
                candidates.append(nums[lo])
            if lo > 0:
                candidates.append(nums[lo-1])
                
            for a in candidates:
                if a == b:
                    continue
                # Calculate error
                # |a/b - 1.5|
                # To avoid floating point issues, we can compare |2a - 3b| / (2b)
                # Minimizing |2a - 3b| / (2b) is equivalent to minimizing |2a - 3b| / b
                # But since b varies, we should compute the actual value.
                err = abs(a / b - target_ratio)
                if err < best_err:
                    best_err = err
                    best_pair = [a, b]
                    
            # Optional: break early if best_err is very small? 
            # Unlikely to be 0.
            
        if time.time() > deadline:
            break

    # Ensure types are correct
    a, b = best_pair
    if not isinstance(a, int): a = int(a)
    if not isinstance(b, int): b = int(b)
    
    return [a, b]
# EVOLVE-BLOCK-END
