import numpy as np
import time
import bisect

def construct():
    start_time = time.time()
    deadline = 85.0
    L = 10_000_000
    
    # 1. Sieve for sigma
    sigma = np.zeros(L + 1, dtype=np.int64)
    for i in range(1, L + 1):
        sigma[i::i] += i
        
    # 2. Group by sigma value
    # Using argsort is faster than a Python dict loop for 10M items
    indices = np.argsort(sigma)
    
    fibers = []
    current_sigma = -1
    current_list = []
    
    for idx in indices:
        n = int(idx)
        s = int(sigma[n])
        if s != current_sigma:
            if current_list:
                fibers.append(current_list)
            current_sigma = s
            current_list = []
        current_list.append(n)
    if current_list:
        fibers.append(current_list)
        
    # 3. Search
    best_error = float('inf')
    best_a = 1
    best_b = 1
    
    # Sort fibers by max element descending? 
    # Larger b allows smaller error potentially. 
    # But error is relative. 
    # Let's just process all.
    
    for lst in fibers:
        if time.time() - start_time > deadline:
            break
        if len(lst) < 2:
            continue
            
        lst.sort()
        
        # Iterate b from largest to smallest
        # If b is small, the denominator 2b is small, so error is larger for same numerator.
        # So large b is better.
        for bi in range(len(lst) - 1, -1, -1):
            b = lst[bi]
            
            # Pruning: 
            # Min possible error for this b is 0 (if a=1.5b in fiber).
            # If best_error is already very small, and b is small, 
            # can we skip?
            # Error >= 1/(2b) is not true, error can be 0.
            # But if 2a-3b != 0, min error is 1/(2b).
            # So if best_error < 1/(2b), then for this b, 
            # if a != 1.5b, error > best_error.
            # If a == 1.5b, error = 0, which is better.
            # So we must check if 1.5b is in the fiber.
            
            target = 1.5 * b
            idx_a = bisect.bisect_left(lst, target)
            
            # Check idx_a and idx_a - 1
            # These are the closest candidates to 1.5b
            candidates = []
            if idx_a < len(lst):
                candidates.append(lst[idx_a])
            if idx_a > 0:
                candidates.append(lst[idx_a - 1])
                
            for a in candidates:
                if a == b:
                    continue
                num = abs(2 * a - 3 * b)
                if num == 0:
                    # Perfect match
                    return [int(a), int(b)]
                err = num / (2.0 * b)
                if err < best_error:
                    best_error = err
                    best_a = a
                    best_b = b
                    
    return [int(best_a), int(best_b)]
