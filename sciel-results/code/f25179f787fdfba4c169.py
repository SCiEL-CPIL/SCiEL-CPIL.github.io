import time
from itertools import combinations
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # Precompute all 4-subsets of [1,16] as bitmasks
    all_sets = []
    for combo in combinations(range(1, 17), 4):
        mask = 0
        for x in combo:
            mask |= (1 << x)
        all_sets.append((mask, combo))
    
    N = len(all_sets)
    masks = [m for m, _ in all_sets]
    
    # Vectorized Precomputation of Intersection Matrix
    # masks_np: (N,) uint32
    # We want I[i][j] = popcount(masks[i] & masks[j])
    
    masks_np = np.array(masks, dtype=np.uint32)
    
    # Create a matrix of pairwise ANDs
    # This creates a (N, N) array of uint32
    # 1820 * 1820 * 4 bytes = ~13 MB. Feasible.
    and_matrix = np.bitwise_and(masks_np[:, None], masks_np[None, :])
    
    # Popcount
    # NumPy >= 1.24 has np.bitwise_count. If not, use a lookup table or bin(x).count('1')
    # Let's try to use a lookup table for 16-bit chunks to be safe and fast, or just assume modern numpy.
    # Actually, a simple loop over the flattened array with .bit_count() is O(N^2) python calls.
    # Better: use a lookup table for 8-bit or 16-bit.
    
    # Lookup table for 8-bit popcount
    popcount_8 = np.array([bin(i).count('1') for i in range(256)], dtype=np.uint8)
    
    # and_matrix is uint32.
    # Split into 4 chunks of 8 bits?
    # m0 = and_matrix & 0xFF
    # m1 = (and_matrix >> 8) & 0xFF
    # m2 = (and_matrix >> 16) & 0xFF
    # m3 = (and_matrix >> 24) & 0xFF
    # popcount = popcount_8[m0] + popcount_8[m1] + popcount_8[m2] + popcount_8[m3]
    
    m0 = and_matrix & 0xFF
    m1 = (and_matrix >> 8) & 0xFF
    m2 = (and_matrix >> 16) & 0xFF
    m3 = (and_matrix >> 24) & 0xFF
    
    inter = popcount_8[m0] + popcount_8[m1] + popcount_8[m2] + popcount_8[m3]
    inter = inter.astype(np.uint8)
    
    # Free memory
    del and_matrix, m0, m1, m2, m3
    
    # Greedy with randomization
    random.seed(42)
    best_family = []
    best_size = 0
    
    # We can run many trials now.
    # Let's run until time runs out.
    
    # Precompute adjacency lists for each intersection size to speed up conflict check?
    # adj[i][s] = list of j such that inter[i][j] == s
    # This might be memory heavy or slow to build.
    # Instead, we just use the inter matrix directly.
    
    num_trials = 0
    while time.time() < deadline:
        num_trials += 1
        indices = list(range(N))
        random.shuffle(indices)
        
        chosen_indices = []
        
        for idx in indices:
            # Check conflict
            conflict = False
            
            # Bucket chosen indices by intersection size with idx
            # buckets[s] = list of chosen indices c such that inter[idx][c] == s
            buckets = [[] for _ in range(4)]
            
            # Optimization: iterate over chosen_indices
            for c_idx in chosen_indices:
                s = inter[idx][c_idx]
                buckets[s].append(c_idx)
                
            # Check for sunflower
            # For each s, if there are at least 2 elements in buckets[s],
            # check if any pair (a, b) in buckets[s] has inter[a][b] == s.
            
            for s in range(4):
                b = buckets[s]
                len_b = len(b)
                if len_b < 2:
                    continue
                
                # Check pairs
                # If len_b is large, this is O(len_b^2).
                # Max len_b is len(chosen_indices).
                for i in range(len_b):
                    i_idx = b[i]
                    for j in range(i+1, len_b):
                        j_idx = b[j]
                        if inter[i_idx][j_idx] == s:
                            conflict = True
                            break
                    if conflict:
                        break
                if conflict:
                    break
            
            if not conflict:
                chosen_indices.append(idx)
                
        current_size = len(chosen_indices)
        if current_size > best_size:
            best_size = current_size
            best_family = [all_sets[i][1] for i in chosen_indices]
            
            # Optional: Local search / improvement
            # Try to swap elements to see if we can add more?
            # For now, just greedy.
            
    return [sorted(s) for s in best_family]
