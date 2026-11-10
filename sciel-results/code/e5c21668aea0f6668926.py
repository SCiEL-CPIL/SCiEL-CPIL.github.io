import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 80.0  # Safety margin for 90s limit
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    p_len = len(primes)
    
    # We need to evaluate n in [1000, 1000000]
    # The numbers involved are n+1 to n+24
    # Max number is 1000000 + 24 = 1000024
    # Min number is 1000 + 1 = 1001
    
    # Precompute masks for all numbers in the relevant range
    # Range: 1001 to 1000024
    min_val = 1001
    max_val = 1000024
    size = max_val - min_val + 1
    
    # masks[i] corresponds to number min_val + i
    masks = np.zeros(size, dtype=np.int32)
    
    # Efficiently compute masks using vectorized operations or loops
    # Given the size (~1M) and 9 primes, a loop over primes is fast enough
    # For each prime p, find multiples in the range and set the bit
    bit_indices = np.arange(p_len)
    
    for i, p in enumerate(primes):
        bit = 1 << i
        # Find first multiple of p >= min_val
        first_multiple = ((min_val + p - 1) // p) * p
        # Find last multiple of p <= max_val
        last_multiple = (max_val // p) * p
        
        if first_multiple > last_multiple:
            continue
            
        # Indices in the masks array
        start_idx = first_multiple - min_val
        end_idx = last_multiple - min_val
        
        # Set the bit for all multiples
        # Using numpy slicing with step
        masks[start_idx : end_idx + 1 : p] |= bit
        
    best_n = 1000
    best_count = 24 # Upper bound is number of primes, but actually max 9
    
    # Sliding window
    # Window size is 24
    # We need to compute OR of masks for windows starting at n
    # Window for n uses masks corresponding to n+1 ... n+24
    # In our array, these are indices (n+1 - min_val) to (n+24 - min_val)
    
    # Initial window for n=1000
    # Indices: 1001-1001=0 to 1024-1001=23
    # Wait, min_val is 1001.
    # n=1000 -> nums 1001..1024 -> indices 0..23
    
    # Compute initial OR
    current_or = 0
    for i in range(24):
        current_or |= masks[i]
        
    current_count = np.bit_count(current_or)
    if current_count < best_count:
        best_count = current_count
        best_n = 1000
        
    # Iterate n from 1001 to 1000000
    # For each step, remove masks[old_start] and add masks[new_end]
    # However, bitwise OR is not invertible. 
    # We need to recompute the OR for each window or use a frequency count of bits.
    # Frequency count is better for sliding window.
    
    # Let's use a bit-count array for the current window
    bit_counts = np.zeros(p_len, dtype=np.int32)
    
    # Initialize bit_counts for n=1000
    for i in range(24):
        m = masks[i]
        # Count set bits per position
        # We can just iterate through bits since p_len is small (9)
        for b in range(p_len):
            if (m >> b) & 1:
                bit_counts[b] += 1
                
    # Number of distinct primes is count of non-zero entries in bit_counts
    current_count = np.count_nonzero(bit_counts)
    if current_count < best_count:
        best_count = current_count
        best_n = 1000
        
    # Slide window
    # n goes from 1001 to 1000000
    # Window for n: indices (n+1-min_val) to (n+24-min_val)
    # Previous window (n-1): indices (n-min_val) to (n+23-min_val)
    # Remove index (n-min_val), Add index (n+24-min_val)
    
    max_n = 1000000
    for n in range(1001, max_n + 1):
        # Check time
        if time.time() > deadline:
            break
            
        # Remove the element leaving the window: n-min_val
        # Wait, for window n-1, the first element is n.
        # Index of n is n - min_val.
        idx_out = n - min_val
        m_out = masks[idx_out]
        for b in range(p_len):
            if (m_out >> b) & 1:
                bit_counts[b] -= 1
                
        # Add the element entering the window: n+24
        # Index of n+24 is n+24 - min_val
        idx_in = n + 24 - min_val
        m_in = masks[idx_in]
        for b in range(p_len):
            if (m_in >> b) & 1:
                bit_counts[b] += 1
                
        # Update count
        # Since p_len is small, np.count_nonzero is fast
        current_count = np.count_nonzero(bit_counts)
        
        if current_count < best_count:
            best_count = current_count
            best_n = n
            
    return best_n
# EVOLVE-BLOCK-END
