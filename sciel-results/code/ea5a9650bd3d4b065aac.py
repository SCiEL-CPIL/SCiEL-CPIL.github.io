import time
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # seconds, safety margin before 90s kill

    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]

    N_MIN = 1000
    N_MAX = 1000000
    WINDOW = 24

    MAX_VAL = N_MAX + WINDOW

    # Compute extracted[m] = product of p^k for p<24 where p^k || m
    # Using sieve: for each prime p, for each power p^k, multiply all multiples of p^k by p
    extracted = np.ones(MAX_VAL + 1, dtype=np.int64)
    for p in small_primes:
        p_power = p
        while p_power <= MAX_VAL:
            extracted[p_power::p_power] *= p
            p_power *= p

    if time.time() - start_time > DEADLINE:
        return N_MIN

    # Sliding window approach using vectorized NumPy operations
    # Window for n covers indices n+1 to n+24 (24 elements)
    # We need to find n that minimizes the number of distinct values in extracted[n+1:n+25]

    # Number of possible window positions
    num_positions = N_MAX - N_MIN + 1  # n ranges from N_MIN to N_MAX

    # For each position n (0-indexed offset from N_MIN), the window is extracted[N_MIN+offset+1 : N_MIN+offset+25]
    # Let's use a different approach: compute distinct counts for all windows at once using a smart method

    # Since we can't easily vectorize "distinct count in sliding window", 
    # let's use a hybrid: process in chunks with NumPy for the inner loop
    # But first, let's try a pure NumPy approach using sorting within windows

    # Actually, let's use an efficient sliding window with NumPy arrays
    # Key insight: we can precompute the extracted values and use a frequency-based approach

    # For better performance, let's use a different strategy:
    # 1. First, get all unique extracted values and map them to smaller indices
    unique_vals, inverse = np.unique(extracted[N_MIN+1:N_MAX+WINDOW+1], return_inverse=True)
    num_unique = len(unique_vals)
    
    # inverse[i] gives the index of extracted[N_MIN+1+i] in unique_vals
    # Window for n = N_MIN + offset: covers indices offset+1 to offset+24 in the original array
    # In the inverse array, that's indices offset to offset+23 (since inverse starts at N_MIN+1)
    
    # For each window of size 24, count distinct values
    # We'll use a sliding window frequency count
    W = WINDOW
    num_windows = num_positions  # number of valid windows
    
    # Create a count array for the first window
    freq = np.zeros(num_unique, dtype=np.int32)
    first_window = inverse[0:W]
    np.add.at(freq, first_window, 1)
    distinct_counts = np.zeros(num_windows, dtype=np.int32)
    distinct_counts[0] = np.sum(freq > 0)
    
    best_count = distinct_counts[0]
    best_n = N_MIN
    
    # Slide the window
    for w in range(1, num_windows):
        if time.time() - start_time > DEADLINE:
            break
            
        # Remove element at position w-1, add element at position w+W-1
        old_idx = inverse[w-1]
        new_idx = inverse[w+W-1]
        
        freq[old_idx] -= 1
        if freq[old_idx] == 0:
            distinct_counts[w] = distinct_counts[w-1] - 1
        else:
            distinct_counts[w] = distinct_counts[w-1]
            
        freq[new_idx] += 1
        if freq[new_idx] == 1:
            distinct_counts[w] += 1
            
        if distinct_counts[w] < best_count:
            best_count = distinct_counts[w]
            best_n = N_MIN + w
            if best_count == 1:
                break

    return best_n
