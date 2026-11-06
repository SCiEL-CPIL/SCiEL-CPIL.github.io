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
    # For fast delta computation, precompute per-index arrays
    # idx_ap_indices[idx] = array of AP indices containing idx
    # idx_ap_positions[idx] = array of positions (0-3) of idx in those APs
    idx_ap_indices = [None] * N
    idx_ap_positions = [None] * N
    
    for idx in range(N):
        ap_indices_list = []
        positions_list = []
        for ap_i in range(num_aps):
            for pos in range(4):
                if aps[ap_i, pos] == idx:
                    ap_indices_list.append(ap_i)
                    positions_list.append(pos)
                    break
        if ap_indices_list:
            idx_ap_indices[idx] = np.array(ap_indices_list, dtype=np.int32)
            idx_ap_positions[idx] = np.array(positions_list, dtype=np.int32)
        else:
            idx_ap_indices[idx] = np.array([], dtype=np.int32)
            idx_ap_positions[idx] = np.array([], dtype=np.int32)

    def get_score(colors):
        c0 = colors[aps[:, 0]]
        c1 = colors[aps[:, 1]]
        c2 = colors[aps[:, 2]]
        c3 = colors[aps[:, 3]]
        is_mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return int(np.sum(is_mono))

    def compute_delta_single(colors, idx):
        ap_idx = idx_ap_indices[idx]
        if len(ap_idx) == 0:
            return 0
            
        positions = idx_ap_positions[idx]
        
        # Get colors for all 4 positions in these APs
        c0 = colors[aps[ap_idx, 0]]
        c1 = colors[aps[ap_idx, 1]]
        c2 = colors[aps[ap_idx, 2]]
        c3 = colors[aps[ap_idx, 3]]
        
        old_mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        old_count = int(np.sum(old_mono))
        
        # Flip the specific positions
        n_c0 = c0.copy()
        n_c1 = c1.copy()
        n_c2 = c2.copy()
        n_c3 = c3.copy()
        
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
        ap_i = idx_ap_indices[i]
        ap_j = idx_ap_indices[j]
        
        if len(ap_i) == 0 and len(ap_j) == 0:
            return 0
            
        if len(ap_i) == 0:
            return compute_delta_single(colors, j)
            
        if len(ap_j) == 0:
            return compute_delta_single(colors, i)
            
        # Union of APs
        ap_union = np.unique(np.concatenate([ap_i, ap_j]))
        
        c0 = colors[aps[ap_union, 0]]
        c1 = colors[aps[ap_union, 1]]
        c2 = colors[aps[ap_union, 2]]
        c3 = colors[aps[ap_union, 3]]
        
        old_mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        old_count = int(np.sum(old_mono))
        
        n_c0 = c0.copy()
        n_c1 = c1.copy()
        n_c2 = c2.copy()
        n_c3 = c3.copy()
        
        # Determine positions of i and j in each AP in the union
        # Use np.searchsorted since ap_union is sorted
        pos_i = np.full(len(ap_union), -1, dtype=np.int32)
        pos_j = np.full(len(ap_union), -1, dtype=np.int32)
        
        # For i: find which APs in union contain i
        # ap_i contains the AP indices for i, idx_ap_positions[i] has positions
        # We need to map each ap in ap_i to its position in ap_union
        search_i = np.searchsorted(ap_union, ap_i)
        # Verify these are valid (ap_i elements should all be in ap_union)
        valid_i = search_i < len(ap_union)
        for k in range(len(ap_i)):
            if valid_i[k]:
                local_idx = search_i[k]
                if ap_union[local_idx] == ap_i[k]:
                    pos_i[local_idx] = idx_ap_positions[i][k]
        
        # For j: find which APs in union contain j
        search_j = np.searchsorted(ap_union, ap_j)
        valid_j = search_j < len(ap_union)
        for k in range(len(ap_j)):
            if valid_j[k]:
                local_idx = search_j[k]
                if ap_union[local_idx] == ap_j[k]:
                    pos_j[local_idx] = idx_ap_positions[j][k]
        
        # Apply flips for i
        mask_i0 = (pos_i == 0)
        mask_i1 = (pos_i == 1)
        mask_i2 = (pos_i == 2)
        mask_i3 = (pos_i == 3)
        
        n_c0[mask_i0] = 1 - n_c0[mask_i0]
        n_c1[mask_i1] = 1 - n_c1[mask_i1]
        n_c2[mask_i2] = 1 - n_c2[mask_i2]
        n_c3[mask_i3] = 1 - n_c3[mask_i3]
        
        # Apply flips for j
        mask_j0 = (pos_j == 0)
        mask_j1 = (pos_j == 1)
        mask_j2 = (pos_j == 2)
        mask_j3 = (pos_j == 3)
        
        n_c0[mask_j0] = 1 - n_c0[mask_j0]
        n_c1[mask_j1] = 1 - n_c1[mask_j1]
        n_c2[mask_j2] = 1 - n_c2[mask_j2]
        n_c3[mask_j3] = 1 - n_c3[mask_j3]
        
        new_mono = (n_c0 == n_c1) & (n_c1 == n_c2) & (n_c2 == n_c3)
        new_count = int(np.sum(new_mono))
        
        return new_count - old_count

    def thue_morse(n):
        """Generate Thue-Morse sequence of length n"""
        result = [0]
        for _ in range(int(np.log2(n)) + 1):
            result = result + [1 - x for x in result]
        return result[:n]

    def mod_pattern(n, period, values):
        """Generate a periodic pattern"""
        return [values[i % period] for i in range(n)]

    best_score = float('inf')
    best_colors = None
    
    # Multiple initialization strategies
    initializations = [
        lambda: np.random.randint(0, 2, N).astype(np.int32),
        lambda: np.array(thue_morse(N), dtype=np.int32),
        lambda: np.array(mod_pattern(N, 4, [0, 0, 1, 1]), dtype=np.int32),
        lambda: np.array(mod_pattern(N, 3, [0, 0, 1]), dtype=np.int32),
        lambda: np.array(mod_pattern(N, 2, [0, 1]), dtype=np.int32),
    ]
    
    num_restarts = 12
    time_per_restart = (deadline - start_time - 5.0) / num_restarts
    
    for restart in range(num_restarts):
        if time.time() >= deadline - 5.0:
            break
            
        restart_deadline = min(start_time + time_per_restart * (restart + 1), deadline - 5.0)
        
        # Choose initialization strategy
        init_idx = restart % len(initializations)
        colors = initializations[init_idx]()
        current_score = get_score(colors)
        
        if current_score < best_score:
            best_score = current_score
            best_colors = colors.copy()
            
        # Simulated Annealing
        T = 15.0
        cooling_factor = 0.998
        min_T = 0.02

        # Probability of double flip
        p_double = 0.30
        
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
                
                # Double flip polish
                if time.time() < restart_deadline - 1.0:
                    for _ in range(100):
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
                for _ in range(200):
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
