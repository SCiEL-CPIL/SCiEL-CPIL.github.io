import time
import numpy as np
import random
from math import sqrt

def _num_divisors(n):
    count = 0
    i = 1
    while i * i <= n:
        if n % i == 0:
            count += 1
            if i != n // i:
                count += 1
        i += 1
    return count

def _greedy_construct(candidates, deadline, max_val=256, max_size=30):
    max_sum = max_size * max_val + 1
    
    current_set = []
    sums = np.zeros(max_sum + 1, dtype=bool)
    sums[0] = True
    
    for x in candidates:
        if time.time() > deadline:
            break
        if len(current_set) >= max_size:
            break
            
        # Quick check: x must not divide any existing sum, and no existing sum must divide x.
        # This is a necessary condition for validity.
        # If x is a multiple of s, s|x. If s is a multiple of x, x|s.
        # We can check this using the sums array.
        # s divides x: any s in sums such that x % s == 0.
        # x divides s: any s in sums such that s % x == 0.
        
        # Since max_sum is ~7680, we can just iterate over the active sums.
        # But getting active sums from numpy array is np.nonzero(sums)[0].
        # This is O(N).
        
        # Let's just do the full check if the quick check passes.
        # Quick check optimization:
        # If x is prime, it's less likely to divide small sums? No.
        
        temp_sums = sums.copy()
        
        if x <= max_sum:
            temp_sums[x:] = temp_sums[x:] | sums[:max_sum + 1 - x]
        
        valid = True
        # Check divisibility
        # We only need to check i up to max_sum // 2
        # And only if temp_sums[i] is True.
        
        # Optimization: Get indices of True values in temp_sums up to max_sum//2
        # Then for each i, check if any multiple is present.
        
        # The previous loop `for i in range(1, max_sum + 1)` is slow in Python.
        # Let's try to vectorize the check or use a faster loop.
        
        # Alternative:
        # For each i in 1..max_sum//2:
        #   if temp_sums[i]:
        #      multiples = 2*i, 3*i, ...
        #      if any(temp_sums[m] for m in multiples): invalid
        
        # Using np.any on slices is fast.
        # Let's stick to the loop but break early.
        
        # We can skip i if temp_sums[i] is False.
        # np.nonzero gives indices.
        active_indices = np.nonzero(temp_sums[:max_sum // 2 + 1])[0]
        
        for i in active_indices:
            # i is at least 1
            # Check multiples
            # Slicing: temp_sums[2*i : max_sum + 1 : i]
            # If i is large, the slice is small.
            if np.any(temp_sums[2*i : max_sum + 1 : i]):
                valid = False
                break
                
        if valid:
            current_set.append(x)
            sums = temp_sums
            
    return current_set

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin
    
    candidates_orders = []
    
    # 1. Descending
    candidates_orders.append(list(range(256, 0, -1)))
    
    # 2. Ascending
    candidates_orders.append(list(range(1, 257)))
    
    # 3. Sorted by number of divisors (ascending - primes first)
    candidates_orders.append(sorted(range(1, 257), key=_num_divisors))
    
    # 4. Sorted by number of divisors (descending - highly composite first)
    candidates_orders.append(sorted(range(1, 257), key=lambda x: -_num_divisors(x)))
    
    # 5. Random permutations
    rng = np.random.default_rng(42)
    for _ in range(5):
        perm = rng.permutation(256) + 1
        candidates_orders.append(perm.tolist())
        
    best_set = []
    
    for order in candidates_orders:
        if time.time() > deadline:
            break
            
        # Run greedy
        current_set = _greedy_construct(order, deadline)
        
        if len(current_set) > len(best_set):
            best_set = current_set
            
        # If we found a set of size 30, we can't do better.
        if len(best_set) >= 30:
            break
            
    return best_set
