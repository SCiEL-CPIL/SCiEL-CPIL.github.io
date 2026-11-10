import numpy as np
import time
from collections import defaultdict
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    MAX_N = 10000000
    
    # 1. Precompute sigma(n) using NumPy sieve
    # This is much faster than pure Python loops.
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    
    # Sieve for sigma
    # for i in range(1, MAX_N + 1):
    #     sigma[i::i] += i
    # This loop in Python is still 10^7 iterations.
    # Each iteration does a NumPy vector addition.
    # The overhead of 10^7 Python loop iterations might be significant (~5-10s).
    # Can we vectorize the loop? Not easily.
    # However, 10^7 simple iterations in Python take ~0.5-1s if the body is simple?
    # No, `sigma[i::i] += i` involves creating a view and adding.
    # Let's estimate: 10^7 iterations * ~1 microsecond per iteration (optimistic) = 10s.
    # This is risky.
    
    # Alternative: Use a standard Python list for sigma and a pure Python sieve?
    # Pure Python sieve:
    # sigma = [0] * (MAX_N + 1)
    # for i in range(1, MAX_N + 1):
    #     for j in range(i, MAX_N + 1, i):
    #         sigma[j] += i
    # This is ~1.6 * 10^8 operations. In Python, this is ~10-20 seconds.
    # NumPy approach: 10^7 outer iterations. Each inner is vectorized.
    # The outer loop is the bottleneck.
    
    # Let's try to optimize the NumPy loop.
    # We can skip even numbers? No, sigma is needed for all.
    
    # Let's stick to NumPy but be careful.
    # If it's too slow, we fall back.
    
    for i in range(1, MAX_N + 1):
        sigma[i::i] += i
        
    if time.time() > deadline - 10:
        # Fallback if sieve is too slow
        return [11, 6]

    # 2. Group numbers by sigma value
    fibers = defaultdict(list)
    # Iterating over numpy array in Python is slow.
    # Convert to list? No, memory.
    # Iterate using range.
    for n in range(1, MAX_N + 1):
        fibers[sigma[n]].append(n)
        
    if time.time() > deadline - 5:
        return [11, 6]

    # 3. Search for best pair
    best_a = 11
    best_b = 6
    best_err = abs(11/6 - 1.5)
    
    # To improve speed, we can sort fibers by max value or just iterate.
    # Most fibers are small.
    
    for s_val, nums in fibers.items():
        if time.time() > deadline - 2:
            break
            
        if len(nums) < 2:
            continue
            
        nums.sort()
        
        # For each b, find a closest to 1.5*b
        # We can optimize by only checking b's that are "large" enough to potentially beat best_err?
        # err = |2a - 3b| / (2b)
        # If 2b * best_err < 1, then |2a - 3b| must be 0 (since integer diff).
        # If 2b * best_err >= 1, we need |2a - 3b| <= floor(2b * best_err).
        
        # Let's just do the binary search.
        for b in nums:
            if time.time() > deadline - 1:
                break
                
            target_a = 1.5 * b
            pos = bisect.bisect_left(nums, target_a)
            
            # Check pos-1 and pos
            # Also pos+1? No, bisect_left returns first index >= target.
            # So nums[pos-1] < target <= nums[pos].
            # Closest are pos-1 and pos.
            
            candidates = []
            if pos > 0:
                candidates.append(nums[pos-1])
            if pos < len(nums):
                candidates.append(nums[pos])
            
            for a in candidates:
                if a == b:
                    continue
                
                # Calculate error exactly or with floats?
                # Floats are fine for comparison, as we return ints.
                err = abs(a / b - 1.5)
                
                if err < best_err:
                    best_err = err
                    best_a = a
                    best_b = b
                    
                    if err == 0.0:
                        return [int(best_a), int(best_b)]

    return [int(best_a), int(best_b)]
