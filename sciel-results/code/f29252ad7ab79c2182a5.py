import time
from math import lcm
from bisect import bisect_left, bisect_right

def construct():
    start_time = time.time()
    deadline = 50.0 # Safety margin
    
    n = 30
    # Calculate LCM of 1..30
    L = 1
    for i in range(1, n + 1):
        L = lcm(L, i)
        
    # Weights w_i = L / i
    # w_i are integers
    weights = [L // i for i in range(1, n + 1)]
    
    # Split into two halves
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    size1 = len(w1)
    size2 = len(w2)
    
    # Generate all sums for first half
    # Store as (sum, code)
    sums1 = []
    # Iterative generation
    # Current sums: list of (sum, code)
    current = [(0, 0)]
    for i in range(size1):
        w = w1[i]
        next_current = []
        for s, c in current:
            # c represents the code for indices 0..i-1
            # We append the digit for index i
            # digit 0 -> -1
            # digit 1 ->  0
            # digit 2 ->  1
            # New code = c * 3 + digit
            next_current.append((s - w, c * 3 + 0))
            next_current.append((s, c * 3 + 1))
            next_current.append((s + w, c * 3 + 2))
        current = next_current
        if time.time() - start_time > deadline:
            break
    sums1 = current
    
    # Generate all sums for second half
    sums2 = []
    current = [(0, 0)]
    for i in range(size2):
        w = w2[i]
        next_current = []
        for s, c in current:
            next_current.append((s - w, c * 3 + 0))
            next_current.append((s, c * 3 + 1))
            next_current.append((s + w, c * 3 + 2))
        current = next_current
        if time.time() - start_time > deadline:
            break
    sums2 = current
    
    # Sort sums2 by sum
    sums2.sort(key=lambda x: x[0])
    
    best_abs_sum = None
    best_code1 = None
    best_code2 = None
    
    # For each sum in sums1, find best matching sum in sums2
    # We want s1 + s2 close to 0
    # s2 close to -s1
    
    # Extract sums2 values for binary search
    s2_values = [s for s, c in sums2]
    
    for s1, c1 in sums1:
        if time.time() - start_time > deadline:
            break
            
        target = -s1
        
        # Find insertion point
        idx = bisect_left(s2_values, target)
        
        # Check idx and idx-1
        candidates = []
        if idx < len(s2_values):
            candidates.append(idx)
        if idx > 0:
            candidates.append(idx - 1)
        
        for i in candidates:
            s2, c2 = sums2[i]
            total = s1 + s2
            if total == 0:
                continue # Exclude zero
            abs_total = abs(total)
            if best_abs_sum is None or abs_total < best_abs_sum:
                best_abs_sum = abs_total
                best_code1 = c1
                best_code2 = c2
    
    if best_abs_sum is None:
        # Fallback to seed
        return [0] * 28 + [1, -1]
        
    # Decode coefficients
    coeffs = [0] * n
    
    # Decode c1 for first half
    # c1 is base-3 encoded, least significant digit corresponds to index 0
    c = best_code1
    for i in range(size1):
        digit = c % 3
        c //= 3
        if digit == 0:
            coeffs[i] = -1
        elif digit == 1:
            coeffs[i] = 0
        else:
            coeffs[i] = 1
            
    # Decode c2 for second half
    c = best_code2
    for i in range(size2):
        digit = c % 3
        c //= 3
        if digit == 0:
            coeffs[mid + i] = -1
        elif digit == 1:
            coeffs[mid + i] = 0
        else:
            coeffs[mid + i] = 1
            
    return coeffs
