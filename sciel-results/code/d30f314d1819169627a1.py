import numpy as np
import time
import random

def kissing_number11() -> np.ndarray:
    d = 11
    start_time = time.time()
    deadline = start_time + 550  # Safety margin
    
    # Standard 220-point solution (Weight 2)
    points_list_220 = []
    for i in range(d):
        for j in range(i + 1, d):
            for s1 in [1, -1]:
                for s2 in [1, -1]:
                    v = [0] * d
                    v[i] = s1
                    v[j] = s2
                    points_list_220.append(v)
    points_220 = np.array(points_list_220, dtype=np.int64)
    
    best_points = points_220
    best_k = 220
    
    # Generate all weight-3 vectors
    all_w3 = []
    for i in range(d):
        for j in range(i + 1, d):
            for k in range(j + 1, d):
                for s1 in [1, -1]:
                    for s2 in [1, -1]:
                        for s3 in [1, -1]:
                            v = [0] * d
                            v[i] = s1
                            v[j] = s2
                            v[k] = s3
                            all_w3.append(v)
                            
    all_w3_arr = np.array(all_w3, dtype=np.int64)
    N = len(all_w3_arr)
    
    # Precompute compatibility masks
    # mask[i] is an integer where bit j is 1 if v[i] dot v[j] <= 1
    # Note: v[i] dot v[i] = 3, which is > 1, so we don't include self in mask?
    # The greedy loop checks `if avail & (1 << i)`, so if i is in avail, it's a candidate.
    # When we select i, we update `avail &= masks[i]`.
    # If masks[i] has bit i set to 0, then i is removed from avail.
    # We need to ensure that for any i, masks[i] has bit i = 0.
    # Dot product v[i] dot v[i] = 3 > 1, so it will be 0. Correct.
    
    masks = [0] * N
    # Compute dot products
    # To speed up, we can compute the full dot product matrix, but N=1320 is small.
    # D = all_w3_arr @ all_w3_arr.T
    # This is 1320x1320. Fast.
    D = all_w3_arr @ all_w3_arr.T
    
    for i in range(N):
        # Get indices where D[i, j] <= 1
        # We want a bitmask
        # Iterate j? Or use numpy?
        # Using numpy to get boolean array, then converting to int bitmask is tricky.
        # Better to just loop j, since N is small.
        m = 0
        row = D[i]
        for j in range(N):
            if row[j] <= 1:
                m |= (1 << j)
        masks[i] = m

    attempts = 0
    # Run greedy search
    while time.time() < deadline - 5:
        attempts += 1
        order = np.random.permutation(N)
        avail = (1 << N) - 1
        selected_indices = []
        
        for i in order:
            if avail & (1 << i):
                selected_indices.append(i)
                avail &= masks[i]
                
        if len(selected_indices) > best_k:
            best_k = len(selected_indices)
            best_points = all_w3_arr[selected_indices]
            
            # Optional: Early stop if we hit a high number?
            # 460 is the target.
            if best_k > 400:
                # We can continue to try to improve, but this is already good.
                pass

    return np.around(best_points).astype(np.int64)
