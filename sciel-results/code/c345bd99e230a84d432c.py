"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np

def construct():
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # We need to check n in [1000, 1000000], so n+i in [1001, 1000024]
    # Precompute smooth parts for range [1001, 1000024]
    low = 1001
    high = 1000024
    size = high - low + 1
    
    # Create array of numbers
    nums = np.arange(low, high + 1, dtype=np.int64)
    
    # Compute smooth parts using valuation counting
    smooth_parts = np.ones(size, dtype=np.int64)
    
    for p in primes:
        # Calculate p-adic valuation for all numbers
        val = np.zeros(size, dtype=np.int32)
        power = p
        while power <= high:
            mask = (nums % power == 0)
            val[mask] += 1
            power *= p
        
        # Multiply smooth_parts by p^val
        smooth_parts *= p ** val

    # Map smooth parts to IDs for faster frequency counting
    unique_vals, inverse = np.unique(smooth_parts, return_inverse=True)
    num_unique = len(unique_vals)
    
    # Convert to Python lists for faster scalar access in the tight loop
    inverse_list = inverse.tolist()
    freq_list = [0] * num_unique
    
    # Initialize window for n=1000
    # Window indices in smooth_parts: 0 to 23
    for i in range(24):
        freq_list[inverse_list[i]] += 1
        
    distinct_count = sum(1 for f in freq_list if f > 0)
    best_count = distinct_count
    best_n = 1000
    
    # Slide window
    for n in range(1001, 1000001):
        # Remove element at index (n-1001)
        out_idx = n - 1001
        id_out = inverse_list[out_idx]
        freq_list[id_out] -= 1
        if freq_list[id_out] == 0:
            distinct_count -= 1
            
        # Add element at index (n-977)
        in_idx = n - 977
        id_in = inverse_list[in_idx]
        if freq_list[id_in] == 0:
            distinct_count += 1
        freq_list[id_in] += 1
        
        if distinct_count < best_count:
            best_count = distinct_count
            best_n = n
            if best_count == 1:
                break  # Can't do better than 1
    
    return best_n

# EVOLVE-BLOCK-END
