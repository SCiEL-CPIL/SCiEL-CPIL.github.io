"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 80.0  # safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24
    
    # We need n in [1000, 1000000], and we look at n+1, ..., n+24
    # So the numbers we factor are in [1001, 1000024]
    total_len = N_MAX - N_MIN + 1  # number of valid n values
    
    # Compute the 23-smooth part (factor composed of primes strictly below 24)
    # for each integer in [1001, 1000024]
    # We'll work on an array of the numbers themselves
    nums = np.arange(1001, 1000025, dtype=np.int64)
    
    # For each number, extract the part divisible only by small primes
    # smooth_part[i] = product of p^e for p in small_primes where p^e divides nums[i]
    # But we want to minimize the number of DISTINCT smooth parts in a window of 24
    
    # Strategy: compute smooth part for each number, encode as ID,
    # then for each window of 24 consecutive numbers, count distinct IDs.
    
    # To speed up: process in chunks to fit in memory and allow early termination
    # But first let's try the full array approach
    
    # Initialize smooth parts as 1 (no small prime factor)
    smooth = np.ones(len(nums), dtype=np.int64)
    
    for p in small_primes:
        # For each number, divide out all factors of p and multiply into smooth
        # We need to track how many times p divides each number
        remaining = nums.copy()
        for _ in range(20):  # enough iterations (2^20 > 10^6)
            mask = remaining % p == 0
            if not np.any(mask):
                break
            smooth[mask] *= p
            remaining[mask] //= p
    
    # Now smooth[i] is the 23-smooth part of nums[i]
    # We need to find n such that the 24 values smooth[n+1-1000], ..., smooth[n+24-1000]
    # have the minimum number of distinct values.
    
    # n corresponds to index n - 1000 in our array (since nums[0] = 1001 = 1000+1)
    # For n, we look at indices (n-1000) to (n-1000+23), i.e., 24 consecutive entries
    
    # Encode smooth parts as compact IDs using factorization into exponents
    # Each smooth part is of the form 2^a * 3^b * 5^c * 7^d * 11^e * 13^f * 17^g * 19^h * 23^i
    # We can encode as a tuple of exponents, but for numpy we need a single integer ID
    
    # Alternative: just use the smooth part value itself as the ID
    # The maximum smooth part for numbers up to 1000024: worst case is the number itself if it's 23-smooth
    # 23-smooth numbers up to 10^6: the largest is at most 10^6, so int64 is fine
    
    # For each window of 24, count distinct values
    # This is expensive for 10^6 windows. Let's use a smarter approach.
    
    # Use the fact that we can compute distinct counts via a sliding window with a frequency table.
    # But in numpy, we can use a trick: for each position, we want the number of distinct values
    # in smooth[i:i+24].
    
    # Approach: Use np.unique on each window would be too slow for 10^6 windows.
    # Instead, let's use a different strategy:
    
    # For each possible set of distinct values, we want to minimize the count.
    # A window has k distinct values if and only if all 24 elements belong to some set of k values.
    
    # Let's try a different approach: use the fact that most numbers will have smooth part = 1
    # (if they have no small prime factors) or a small smooth part.
    
    # Let's compute the distinct count for each window using a frequency-based sliding window.
    # We'll use a dictionary-based approach but vectorize where possible.
    
    # Actually, let's try a simpler approach first: since the smooth parts are integers,
    # we can use np.unique with a sliding window approach using stride tricks.
    
    # Create a 2D view of the smooth array with windows of size 24
    # Number of windows = len(smooth) - 24 + 1 = 1000024 - 24 + 1 = 1000001
    # But we only care about windows starting at index 0 to total_len-1 = 999000
    # (since n goes from 1000 to 1000000, index n-1000 goes from 0 to 999000)
    
    # Let's use a chunked approach to avoid memory issues
    chunk_size = 100000
    
    best_n = N_MIN
    best_distinct = float('inf')
    
    # For efficiency, let's process in chunks and track the best
    for chunk_start in range(0, total_len, chunk_size):
        if time.time() > deadline - 5:
            break
        
        chunk_end = min(chunk_start + chunk_size, total_len)
        # Window indices: chunk_start to chunk_end-1
        # Each window i covers smooth[i:i+24]
        # We need smooth indices from chunk_start to chunk_end-1+23
        
        # Extract the relevant slice
        slice_start = chunk_start
        slice_end = chunk_end + WINDOW - 1
        if slice_end > len(smooth):
            slice_end = len(smooth)
        
        window_data = smooth[slice_start:slice_end]
        n_windows = len(window_data) - WINDOW + 1
        
        # For each window, count distinct values
        # Use a simple approach: for each window, use np.unique
        # This is O(n_windows * 24 * log(24)) which should be fast enough
        
        distinct_counts = np.empty(n_windows, dtype=np.int32)
        for i in range(n_windows):
            if time.time() > deadline - 2:
                break
            distinct_counts[i] = len(np.unique(window_data[i:i+WINDOW]))
        
        # Find the minimum in this chunk
        min_idx = np.argmin(distinct_counts[:n_windows])
        min_val = distinct_counts[min_idx]
        
        if min_val < best_distinct:
            best_distinct = min_val
            best_n = N_MIN + chunk_start + min_idx
    
    return best_n
# EVOLVE-BLOCK-END
