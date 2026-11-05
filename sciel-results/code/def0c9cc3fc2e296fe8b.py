import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    min_n = 1000
    max_n = 1000000
    max_val = max_n + 24  # 1000024
    
    # Precompute small prime part using sieve-like approach
    # S[m] = product of p^k where p^k || m for all primes p < 24
    S = np.ones(max_val + 1, dtype=np.int64)
    
    for p in primes:
        pk = p
        while pk <= max_val:
            if time.time() > deadline - 10:
                break
            # For all multiples of pk, multiply S by p
            # This correctly accumulates the full power of p in each number
            S[pk::pk] *= p
            pk *= p
        if time.time() > deadline - 10:
            break
    
    if time.time() > deadline - 10:
        # Fallback if precomputation took too long
        return min_n

    # Sliding window search with vectorized distinct count
    # We want to find n in [min_n, max_n] minimizing distinct values in S[n+1 : n+25]
    
    best_count = 25 # Max possible is 24
    best_n = min_n
    
    # Process in chunks to allow deadline checking and memory management
    # Window size is 24
    # For a chunk of n's from start to end, we need S[start+1 : end+25]
    # We can reshape this into (chunk_size, 24)
    
    chunk_size = 100000
    
    for start in range(min_n, max_n + 1, chunk_size):
        end = min(start + chunk_size, max_n + 1)
        if time.time() > deadline - 2:
            break
        
        # Extract the block of values needed for windows starting at 'start' to 'end-1'
        # Window for n is S[n+1] ... S[n+24]
        # So for n in [start, end-1], we need S[start+1] to S[end-1+24] = S[end+23]
        # Slice: S[start+1 : end+24]
        # Length: (end+24) - (start+1) = end - start + 23
        
        block = S[start+1 : end+24]
        
        # Reshape into (num_windows, 24)
        # num_windows = end - start
        # The block has length (end - start) + 23
        # We need to extract 24 consecutive elements for each window.
        # This is a sliding window view.
        # We can use np.lib.stride_tricks.sliding_window_view or manual slicing if available/fast.
        # Since we are on a specific block, we can construct the 2D array by taking columns.
        
        # Actually, simpler:
        # The block is S[start+1], S[start+2], ..., S[end+23]
        # Window 0 (n=start): S[start+1]...S[start+24] -> indices 0..23 of block
        # Window 1 (n=start+1): S[start+2]...S[start+25] -> indices 1..24 of block
        # ...
        # Window k (n=start+k): indices k..k+23
        
        # We can create a 2D array using strides or just slicing in a loop? No, we want vectorized.
        # Let's use np.lib.stride_tricks.sliding_window_view if available, otherwise manual.
        # But standard numpy doesn't have sliding_window_view until 1.20.
        # Let's assume a compatible environment or use a safe method.
        
        # Safe method:
        # Create an array of indices? No, that's slow.
        # Let's use the fact that we can just sort and count.
        # We can create the 2D array by taking slices of the 1D block.
        # But doing it in a loop is what we want to avoid.
        
        # Alternative: Use np.unique on the whole block? No, we need per-window counts.
        
        # Let's try a different vectorization:
        # For each window, we sort the 24 elements.
        # We can do this by reshaping the block into a structure that allows sorting rows.
        
        # Let's construct the 2D array explicitly using strides if possible, or just a simple loop if chunk is small?
        # No, 100k chunks is good.
        
        # Let's use a trick:
        # We can create a 2D array `windows` of shape (num_windows, 24)
        # windows[i, j] = block[i + j]
        
        # We can do this with:
        # rows = np.arange(num_windows)[:, None]
        # cols = np.arange(24)[None, :]
        # windows = block[rows + cols]
        
        num_windows = end - start
        rows = np.arange(num_windows)[:, None]
        cols = np.arange(24)[None, :]
        windows = block[rows + cols]
        
        # Now sort each row
        sorted_windows = np.sort(windows, axis=1)
        
        # Count distinct values
        # Distinct count = 1 + number of non-zero differences between adjacent elements
        diffs = np.diff(sorted_windows, axis=1)
        distinct_counts = 1 + np.sum(diffs != 0, axis=1)
        
        # Find min in this chunk
        min_idx_in_chunk = np.argmin(distinct_counts)
        min_count_in_chunk = distinct_counts[min_idx_in_chunk]
        
        if min_count_in_chunk < best_count:
            best_count = min_count_in_chunk
            best_n = start + min_idx_in_chunk
            
            if best_count == 1:
                return best_n

    return best_n
