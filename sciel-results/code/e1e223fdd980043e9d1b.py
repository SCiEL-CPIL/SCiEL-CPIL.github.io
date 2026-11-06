import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    N = 256
    
    # Precompute all 4-term arithmetic progressions
    ap_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(1, max_a + 1):
            ap_list.append((a-1, a+d-1, a+2*d-1, a+3*d-1))
            
    if not ap_list:
        return [0] * N
        
    aps = np.array(ap_list, dtype=np.int32)
    num_aps = len(aps)
    
    # Precompute for each index, which APs contain it and at what position
    # index_aps[idx] = array of shape (k, 2) where column 0 is ap_index, column 1 is pos (0-3)
    index_aps = [[] for _ in range(N)]
    for i in range(num_aps):
        for j in range(4):
            idx = aps[i, j]
            index_aps[idx].append((i, j))
            
    index_aps_arr = []
    for i in range(N):
        if index_aps[i]:
            arr = np.array(index_aps[i], dtype=np.int32)
            index_aps_arr.append(arr)
        else:
            index_aps_arr.append(np.array([], dtype=np.int32).reshape(0, 2))

    def get_score(colors):
        c0 = colors[aps[:, 0]]
        c1 = colors[aps[:, 1]]
        c2 = colors[aps[:, 2]]
        c3 = colors[aps[:, 3]]
        is_mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return int(np.sum(is_mono))

    def compute_delta_single(colors, idx):
        pairs = index_aps_arr[idx]
        if len(pairs) == 0:
            return 0
            
        ap_indices = pairs[:, 0]
        positions = pairs[:, 1]
        
        # Get colors for all 4 positions in these APs
        c0 = colors[aps[ap_indices, 0]]
        c1 = colors[aps[ap_indices, 1]]
        c2 = colors[aps[ap_indices, 2]]
        c3 = colors[aps[ap_indices, 3]]
        
        old_mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        old_count = int(np.sum(old_mono))
        
        # Flip the specific positions
        # We need to create a copy to modify
        n_c0 = c0.copy()
        n_c1 = c1.copy()
        n_c2 = c2.copy()
        n_c3 = c3.copy()
        
        # Flip where position matches
        # positions is an array of the position of idx in each AP
        mask0 = (positions == 0)
        mask1 = (positions == 1)
        mask2 = (positions == 2)
        mask3 = (positions == 3)
        
        n_c0[mask0] = 1 - n_c0[mask0]
        n_c1[mask1] = 1 - n_c1[mask1]
        n_c2[mask2] = 1 - n_c2[mask2]
        n_c3[mask3] = 1 - n_c3[mask3]
        
        new_mono = (n_c0 == n_c1) & (n_c1 == n_c2) & (n_c2 == n_c3)
        new_count = int(np.sum(new_mono))
        
        return new_count - old_count

    def compute_delta_double(colors, i, j):
        pairs_i = index_aps_arr[i]
        pairs_j = index_aps_arr[j]
        
        if len(pairs_i) == 0 and len(pairs_j) == 0:
            return 0
            
        if len(pairs_i) == 0:
            ap_indices = pairs_j[:, 0]
            pos_i_mask = np.array([], dtype=bool)
            pos_j_mask = pairs_j[:, 1]
            # In this case, i is not in any AP, so only j flips
            # But wait, if i is not in any AP, flipping i doesn't change score?
            # Yes, if degree is 0, delta is 0 for i.
            # So this reduces to single flip of j.
            return compute_delta_single(colors, j)
            
        if len(pairs_j) == 0:
            return compute_delta_single(colors, i)
            
        # Union of APs
        ap_indices = np.unique(np.concatenate([pairs_i[:, 0], pairs_j[:, 0]]))
        
        c0 = colors[aps[ap_indices, 0]]
        c1 = colors[aps[ap_indices, 1]]
        c2 = colors[aps[ap_indices, 2]]
        c3 = colors[aps[ap_indices, 3]]
        
        old_mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        old_count = int(np.sum(old_mono))
        
        n_c0 = c0.copy()
        n_c1 = c1.copy()
        n_c2 = c2.copy()
        n_c3 = c3.copy()
        
        # We need to flip i and j in their respective positions within these APs
        # Create masks for which APs in ap_indices contain i and j
        
        # Find which rows in pairs_i correspond to ap_indices
        # pairs_i is (k_i, 2), ap_indices is (m,).
        # We can use np.in1d or create a mapping.
        
        # Efficient way: 
        # Create a set of ap_indices for i and j
        set_i = set(pairs_i[:, 0].tolist())
        set_j = set(pairs_j[:, 0].tolist())
        
        # For each AP in ap_indices, determine if i or j is in it and at what position
        # This loop is over m (num unique APs), which is small (< 50 usually)
        
        # However, we want to vectorize.
        # Let's just iterate over the unique APs if the count is small, 
        # or use advanced indexing logic.
        
        # Given N=256, max degree is ~40. Union is < 80.
        # A simple loop over the union APs might be faster than complex vectorized masking
        # if we can't easily map local indices to global.
        
        # Let's try a vectorized approach:
        # Create arrays for the position of i and j in each AP in ap_indices.
        # pos_i_in_ap[ap_idx] = position of i in ap_idx, or -1 if not present.
        
        pos_i_map = np.full(len(ap_indices), -1, dtype=np.int32)
        pos_j_map = np.full(len(ap_indices), -1, dtype=np.int32)
        
        # Map ap_index to local index in ap_indices array
        # This is O(m^2) if done naively, but m is small.
        # Or use np.searchsorted if sorted. ap_indices is sorted by np.unique.
        
        # Let's just use a dictionary for mapping ap_index -> local_index
        ap_to_local = {ap: idx for idx, ap in enumerate(ap_indices)}
        
        for ap_idx, pos in pairs_i:
            if ap_idx in ap_to_local:
                pos_i_map[ap_to_local[ap_idx]] = pos
                
        for ap_idx, pos in pairs_j:
            if ap_idx in ap_to_local:
                pos_j_map[ap_to_local[ap_idx]] = pos
                
        # Now apply flips
        # Flip i
        mask_i0 = (pos_i_map == 0)
        mask_i1 = (pos_i_map == 1)
        mask_i2 = (pos_i_map == 2)
        mask_i3 = (pos_i_map == 3)
        
        n_c0[mask_i0] = 1 - n_c0[mask_i0]
        n_c1[mask_i1] = 1 - n_c1[mask_i1]
        n_c2[mask_i2] = 1 - n_c2[mask_i2]
        n_c3[mask_i3] = 1 - n_c3[mask_i3]
        
        # Flip j
        mask_j0 = (pos_j_map == 0)
        mask_j1 = (pos_j_map == 1)
        mask_j2 = (pos_j_map == 2)
        mask_j3 = (pos_j_map == 3)
        
        n_c0[mask_j0] = 1 - n_c0[mask_j0]
        n_c1[mask_j1] = 1 - n_c1[mask_j1]
        n_c2[mask_j2] = 1 - n_c2[mask_j2]
        n_c3[mask_j3] = 1 - n_c3[mask_j3]
        
        new_mono = (n_c0 == n_c1) & (n_c1 == n_c2) & (n_c2 == n_c3)
        new_count = int(np.sum(new_mono))
        
        return new_count - old_count

    best_score = float('inf')
    best_colors = None
    
    num_restarts = 15
    time_per_restart = (deadline - start_time - 5.0) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() >= deadline - 5.0:
            break
            
        restart_deadline = min(start_time + time_per_restart * (restart + 1), deadline - 5.0)
        
        # Random initial coloring
        colors = np.random.randint(0, 2, N).astype(np.int32)
        current_score = get_score(colors)
        
        if current_score < best_score:
            best_score = current_score
            best_colors = colors.copy()
            
        # Simulated Annealing with higher initial temperature
        T = 20.0
        cooling_factor = 0.997
        min_T = 0.05

        # Probability of double flip
        p_double = 0.35
        
        while time.time() < restart_deadline and T > min_T:
            if random.random() < p_double:
                i = np.random.randint(0, N)
                j = np.random.randint(0, N)
                if i == j:
                    j = (j + 1) % N
                delta = compute_delta_double(colors, i, j)
                if delta <= 0 or random.random() < np.exp(-delta / T):
                    colors[i] = 1 - colors[i]
                    colors[j] = 1 - colors[j]
                    current_score += delta
            else:
                idx = np.random.randint(0, N)
                delta = compute_delta_single(colors, idx)
                if delta <= 0 or random.random() < np.exp(-delta / T):
                    colors[idx] = 1 - colors[idx]
                    current_score += delta
                    
            if current_score < best_score:
                best_score = current_score
                best_colors = colors.copy()
                
            T *= cooling_factor
            
        # Greedy Polish
        if time.time() < restart_deadline - 1.0:
            improved = True
            while improved and time.time() < restart_deadline - 1.0:
                improved = False
                # Single flip polish
                for idx in range(N):
                    if time.time() >= restart_deadline - 1.0:
                        break
                    delta = compute_delta_single(colors, idx)
                    if delta < 0:
                        colors[idx] = 1 - colors[idx]
                        current_score += delta
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_colors = colors.copy()
                
                # Double flip polish (limited)
                if time.time() < restart_deadline - 1.0:
                    # Try random double flips
                    for _ in range(80):
                        if time.time() >= restart_deadline - 1.0:
                            break
                        i = np.random.randint(0, N)
                        j = np.random.randint(0, N)
                        if i == j:
                            continue
                        delta = compute_delta_double(colors, i, j)
                        if delta < 0:
                            colors[i] = 1 - colors[i]
                            colors[j] = 1 - colors[j]
                            current_score += delta
                            improved = True
                            if current_score < best_score:
                                best_score = current_score
                                best_colors = colors.copy()

    # Final global polish on best solution
    if best_colors is not None and time.time() < deadline - 2.0:
        colors = best_colors.copy()
        current_score = get_score(colors)
        
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for idx in range(N):
                if time.time() >= deadline - 2.0:
                    break
                delta = compute_delta_single(colors, idx)
                if delta < 0:
                    colors[idx] = 1 - colors[idx]
                    current_score += delta
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_colors = colors.copy()
                        
            # Try some double flips
            if time.time() < deadline - 2.0:
                for _ in range(150):
                    if time.time() >= deadline - 2.0:
                        break
                    i = np.random.randint(0, N)
                    j = np.random.randint(0, N)
                    if i == j:
                        continue
                    delta = compute_delta_double(colors, i, j)
                    if delta < 0:
                        colors[i] = 1 - colors[i]
                        colors[j] = 1 - colors[j]
                        current_score += delta
                        improved = True
                        if current_score < best_score:
                            best_score = current_score
                            best_colors = colors.copy()

    if best_colors is None:
        return [0] * N
        
    return [int(c) for c in best_colors]
