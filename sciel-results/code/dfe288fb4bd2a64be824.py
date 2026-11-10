import sys
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 60.0
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    max_n = 1000000
    max_val = max_n + 24
    size = max_val + 1
    
    # Initialize small_part array
    # Using numpy for faster initialization and potential vectorization later
    # But the sieve logic is easier in Python lists or array module for random access modification?
    # Actually, numpy is fine.
    small_part = np.ones(size + 1, dtype=np.int64)
    
    # Sieve-like computation
    # We need to multiply small_part[j] by p for each power of p in j
    # This is not easily vectorizable in a simple loop because of the while loop.
    # However, the range is 1e6, which is small enough for a Python loop if optimized.
    # Let's stick to a fast Python loop for the sieve part, or use a list.
    # List is faster than numpy for element-wise updates in a tight loop with conditionals?
    # Let's use a list for the sieve, then convert to numpy for the window analysis.
    
    small_part_list = [1] * (size + 1)
    
    for p in primes:
        for j in range(p, size + 1, p):
            val = j
            # Remove all factors of p and accumulate
            # Optimization: pre-compute powers? No, while loop is short.
            while val % p == 0:
                val //= p
                small_part_list[j] *= p
                
    # Convert to numpy array for fast window operations
    sp = np.array(small_part_list, dtype=np.int64)
    
    # We need to analyze windows of size 24 for n in [1000, 1000000]
    # Window for n is sp[n+1 : n+25]
    
    # Use sliding_window_view
    # Shape: (num_windows, 24)
    # num_windows = max_n - 1000 + 1
    # Start index for n=1000 is 1001.
    # End index for n=1000000 is 1000024.
    
    window_size = 24
    start_idx = 1001
    end_idx = max_n + 24 # exclusive end for the last window's end element?
    # n+24 is the last element.
    # For n=1000000, last element is 1000024.
    # So we need sp[1001 : 1000025]
    
    # Create the view
    # Note: sliding_window_view requires the array to be contiguous.
    windows = np.lib.stride_tricks.sliding_window_view(sp[start_idx:end_idx+1], window_size)
    
    # Check time
    if time.time() - start_time > timeout - 10.0:
         # Fallback to simple scan if something went wrong or is too slow
         pass

    # Compute max and min for each window
    w_max = np.max(windows, axis=1)
    w_min = np.min(windows, axis=1)
    w_range = w_max - w_min
    
    # We want to find windows with small distinct count.
    # A small range implies a small distinct count.
    # Get the indices of the smallest ranges.
    # Let's take the top 10000 candidates.
    num_candidates = 10000
    if len(w_range) < num_candidates:
        num_candidates = len(w_range)
        
    # argpartition returns indices of the smallest num_candidates elements
    # It's not sorted, but we don't need them sorted, just a subset of the smallest.
    # Actually, we want the absolute smallest, so we should check the very smallest ones first.
    # Let's get the indices of the smallest 5000, sort them by range, and check the top ones.
    
    # To be safe, let's get the smallest 20000, then sort them.
    check_count = min(20000, len(w_range))
    
    if check_count > 0:
        # Get indices of the smallest check_count ranges
        partition_indices = np.argpartition(w_range, check_count - 1)[:check_count]
        
        # Sort these indices by their actual range value to check the smallest ones first
        sorted_indices = partition_indices[np.argsort(w_range[partition_indices])]
        
        best_n = 1000
        best_count = float('inf')
        
        # Iterate through the sorted candidates
        # We can stop if we find count 1 or 2?
        # Min distinct is 1.
        
        for idx in sorted_indices:
            # idx corresponds to the window starting at start_idx + idx
            # n = 1000 + idx
            n = 1000 + idx
            
            # Extract the window values
            # windows[idx] is a view of the 24 values
            window_vals = windows[idx]
            
            # Count distinct
            # Converting to set is fast for 24 elements
            count = len(set(window_vals.tolist()))
            
            if count < best_count:
                best_count = count
                best_n = n
                if best_count == 1:
                    break
                if best_count == 2:
                    # We can continue to see if we find 1, but 2 is very good.
                    # Let's keep searching a bit more, but prioritize speed.
                    pass
            
            # Time check
            if time.time() - start_time > timeout - 2.0:
                break
                
    else:
        best_n = 1000
        best_count = float('inf')
        # Fallback scan if numpy failed
        for n in range(1000, max_n + 1):
            window = small_part_list[n+1 : n+25]
            count = len(set(window))
            if count < best_count:
                best_count = count
                best_n = n
            if time.time() - start_time > timeout - 2.0:
                break

    return best_n
# EVOLVE-BLOCK-END
