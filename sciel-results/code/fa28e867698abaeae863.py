import time
import numpy as np
from fractions import Fraction
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    N = 10000000

    # 1. Compute sigma(n) using vectorized sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve of Sigma
    # This is O(N log N) but vectorized
    for i in range(1, N + 1):
        if time.time() > deadline - 20.0:
            # Fallback if sieve takes too long (unlikely with N=10^7 on modern hardware, but safe)
            return [11, 6]
        sigma[i::i] += i

    # 2. Prepare for grouping
    # We want to process numbers in order of their sigma value to find groups easily.
    # Sorting N=10^7 integers is fast with numpy.
    order = np.argsort(sigma, kind='stable')
    
    # 3. Iterate through sorted order to find groups of equal sigma
    # best_error is represented as Fraction for exact comparison
    best_error = Fraction(1, 3) # Initial guess, error for 11/6 is |11/6 - 1.5| = |11/6 - 9/6| = 2/6 = 1/3
    best_pair = (11, 6)
    
    i = 0
    n_order = len(order)
    
    while i < n_order:
        if time.time() > deadline - 5.0:
            break
            
        # Find the extent of the current group
        current_sigma = sigma[order[i]]
        j = i + 1
        # Since order is sorted by sigma, all elements with the same sigma are contiguous
        while j < n_order and sigma[order[j]] == current_sigma:
            j += 1
            
        group_size = j - i
        if group_size >= 2:
            # Extract the actual numbers in this group
            # order[i:j] contains the indices n such that sigma[n] == current_sigma
            # We need the values themselves, which are the indices in 'order'
            group_vals = order[i:j]
            
            # To minimize |a/b - 3/2|, for each b in group_vals, we want a close to 1.5*b
            # group_vals is sorted because 'order' is sorted? 
            # Wait, np.argsort sorts by value. The indices in 'order' are not necessarily sorted.
            # However, the values group_vals are the numbers n. 
            # Let's sort group_vals to use bisect.
            group_vals.sort()
            
            # For each b in group_vals, find best a
            # We can optimize by noting that a and b are in the same set.
            # |a/b - 1.5| = |2a - 3b| / (2b)
            # We want to minimize this.
            
            for b in group_vals:
                # Target a is 1.5 * b
                target = 1.5 * b
                pos = bisect.bisect_left(group_vals, target)
                
                # Check pos and pos-1
                for idx in [pos - 1, pos]:
                    if 0 <= idx < group_size:
                        a = group_vals[idx]
                        if a == b:
                            continue
                        
                        # Calculate error exactly
                        # err = |a/b - 3/2| = |2a - 3b| / 2b
                        num = abs(2 * a - 3 * b)
                        if num == 0:
                            # Perfect match, error 0. But task says raw value 0 is invalid.
                            # However, if a/b == 3/2, then 2a=3b. 
                            # If we find this, it's the best possible non-zero? No, 0 is invalid.
                            # The task says "A raw value of exactly 0 cannot be scored and counts as invalid."
                            # So we must avoid a/b == 1.5 exactly.
                            # If num == 0, we skip this pair.
                            continue
                        
                        den = 2 * b
                        err = Fraction(num, den)
                        
                        if err < best_error:
                            best_error = err
                            best_pair = (a, b)
                            
        i = j

    return [int(best_pair[0]), int(best_pair[1])]
