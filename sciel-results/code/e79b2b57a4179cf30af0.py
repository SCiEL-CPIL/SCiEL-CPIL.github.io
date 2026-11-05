import numpy as np
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    MAX_N = 1000000
    max_val = MAX_N + 24
    smooth = np.ones(max_val + 1, dtype=np.int64)
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    for p in primes:
        power = p
        while power <= max_val:
            smooth[power::power] *= p
            power *= p
            
    # Window region: smooth[1001] to smooth[1000024] inclusive
    # n in [1000, 1000000], window = smooth[n+1:n+25]
    # Window start indices: 1001 to 1000001
    # Window region length: 1000024 - 1001 + 1 = 999024
    window_region = smooth[1001:1000025]  # length 999024
    n_local = len(window_region)
    n_windows = 999001  # number of windows of size 24
    
    # Compute prev_occurrence for each position using vectorized approach
    # First, get unique values and their first occurrence
    unique_vals, first_occ = np.unique(window_region, return_index=True)
    
    # Map each value to its index in unique_vals
    val_to_idx = {v: i for i, v in enumerate(unique_vals)}
    
    # For each position, find the previous occurrence of the same value
    # We can do this by processing positions in order and tracking last seen
    prev_occ = np.full(n_local, -1, dtype=np.int64)
    
    # Use a dictionary for efficient lookup
    last_seen = {}
    for i in range(n_local):
        val = window_region[i]
        if val in last_seen:
            prev_occ[i] = last_seen[val]
        last_seen[val] = i
    
    # Difference array approach
    # Position i contributes to windows s where max(i-23, prev_occ[i]+1) <= s <= i
    diff = np.zeros(n_windows + 1, dtype=np.int32)
    
    for i in range(n_local):
        s_min = i - 23
        if s_min < 0:
            s_min = 0
        po = prev_occ[i]
        if po + 1 > s_min:
            s_min = po + 1
        s_max = i
        if s_max >= n_windows:
            s_max = n_windows - 1
        if s_min <= s_max:
            diff[s_min] += 1
            diff[s_max + 1] -= 1
    
    distinct_counts = np.cumsum(diff[:n_windows])
    
    min_count = np.min(distinct_counts)
    min_indices = np.where(distinct_counts == min_count)[0]
    best_n = 1000 + int(min_indices[0])
    
    return best_n
