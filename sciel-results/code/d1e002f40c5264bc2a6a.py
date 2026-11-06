"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill

    def time_remaining():
        return DEADLINE - (time.time() - start_time)

    def time_exceeded():
        return time.time() - start_time >= DEADLINE

    # Step 1: Compute phi for a range [1, N] using sieve method
    # We need to find long runs of consecutive integers with distinct totients.
    # The seed program searches up to 20000 and finds best run.
    # Let's use a larger range with sieve for speed.
    
    # Use numpy sieve for phi computation
    N = 200000  # Search up to 200000
    
    # Sieve of Euler for phi
    phi_arr = np.arange(N + 1, dtype=np.int64)
    for i in range(2, N + 1):
        if phi_arr[i] == i:  # i is prime
            for j in range(i, N + 1, i):
                phi_arr[j] -= phi_arr[j] // i
    
    # Now phi_arr[x] = phi(x) for x in 0..N
    # phi(0) = 0, phi(1) = 1
    
    # Step 2: Find longest run of consecutive integers with distinct phi values
    # For each start position s, extend as far as possible while phi values are distinct
    # Use a sliding window / hash set approach
    
    best_k = 1
    best_s = 1
    
    # Efficient approach: for each position, track the maximum length of a valid run ending at that position
    # Or: iterate start positions and use a set
    
    # More efficient: use the fact that if we process left to right, we can maintain a window
    # where all phi values are distinct. When a duplicate is found, we shrink from the left.
    
    # Sliding window approach:
    # Maintain a window [left, right] where all phi values are distinct.
    # For each right, extend as far as possible, then record the window length.
    
    last_seen = {}  # phi_value -> last index where it appeared
    left = 1
    current_max = 1
    
    for right in range(1, N + 1):
        v = phi_arr[right]
        if v in last_seen and last_seen[v] >= left:
            # Duplicate found, move left past the previous occurrence
            left = last_seen[v] + 1
        last_seen[v] = right
        window_len = right - left + 1
        if window_len > current_max:
            current_max = window_len
            best_k = window_len
            best_s = left
    
    # Check if we can do better with a larger search range
    # But first, let's see how long we have
    if time_exceeded():
        return {'start': best_s, 'length': best_k}
    
    # Try a larger range if we have time
    # The key insight: longer runs of distinct phi values are rare and grow slowly
    # Let's try to search in a larger range with a smarter approach
    
    # Actually, let me reconsider. The sliding window finds the longest contiguous
    # subarray where all elements are distinct. This is exactly what we need.
    # But we need s + k - 1 <= 10^12, which is easily satisfied for N up to ~10^12.
    # However, computing phi for such large N is infeasible.
    
    # Let's try to search in multiple ranges or a much larger range.
    # For now, let's try N = 1000000 if time permits.
    
    elapsed = time.time() - start_time
    if elapsed < 100.0:
        # Try larger range
        N2 = 1000000
        phi_arr2 = np.arange(N2 + 1, dtype=np.int64)
        for i in range(2, N2 + 1):
            if phi_arr2[i] == i:
                for j in range(i, N2 + 1, i):
                    phi_arr2[j] -= phi_arr2[j] // i
        
        last_seen2 = {}
        left2 = 1
        current_max2 = 1
        
        for right in range(1, N2 + 1):
            v = phi_arr2[right]
            if v in last_seen2 and last_seen2[v] >= left2:
                left2 = last_seen2[v] + 1
            last_seen2[v] = right
            window_len = right - left2 + 1
            if window_len > current_max2:
                current_max2 = window_len
                best_k = window_len
                best_s = left2
        
        if time_exceeded():
            return {'start': best_s, 'length': best_k}
    
    # Ensure constraints are met
    if best_s + best_k - 1 > 10**12:
        best_k = 10**12 - best_s + 1
    if best_k > 200:
        best_k = 200
    if best_k < 1:
        best_k = 1
    if best_s < 1:
        best_s = 1
    
    return {'start': best_s, 'length': best_k}
# EVOLVE-BLOCK-END
