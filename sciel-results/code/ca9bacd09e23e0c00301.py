import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 145  # safety margin before 150s kill
    
    # Precompute all 4-term arithmetic progressions as numpy arrays
    # A 4-AP: a, a+d, a+2d, a+3d with a >= 1, a+3d <= N, d >= 1
    ap_list = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            ap_list.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    total_aps = len(ap_list)
    # Shape: (total_aps, 4)
    aps = np.array(ap_list, dtype=np.int32)
    
    # For each position, which APs contain it (for delta computation)
    pos_to_aps = [[] for _ in range(N)]
    for i in range(total_aps):
        for j in range(4):
            pos_to_aps[aps[i, j]].append(i)
    
    # Precompute for each position: the AP indices and the position within that AP
    pos_ap_indices = []
    pos_ap_positions = []
    for p in range(N):
        idxs = np.array(pos_to_aps[p], dtype=np.int32)
        positions = np.array([list(aps[i]).index(p) for i in pos_to_aps[p]], dtype=np.int8)
        pos_ap_indices.append(idxs)
        pos_ap_positions.append(positions)
    
    def count_mono(colors):
        """Count monochromatic 4-APs using numpy. colors is numpy array of 0/1."""
        c = colors[aps]  # shape (total_aps, 4)
        mono = (c.max(axis=1) == c.min(axis=1))
        return int(mono.sum())
    
    def fast_delta(colors, p):
        """Fast delta computation using vectorized numpy operations."""
        old_color = colors[p]
        ap_idxs = pos_ap_indices[p]
        n_aps = len(ap_idxs)
        if n_aps == 0:
            return 0
        
        sub_aps = aps[ap_idxs]  # shape (n_aps, 4)
        positions = pos_ap_positions[p]  # shape (n_aps,)
        
        # Get colors of all positions in these APs
        sub_colors = colors[sub_aps]  # shape (n_aps, 4)
        
        # Before flip: mono if all 4 same
        mono_before = (sub_colors.max(axis=1) == sub_colors.min(axis=1))
        
        # After flip: replace position p's color with 1 - old_color
        # Create modified copy
        sub_colors_after = sub_colors.copy()
        rows = np.arange(n_aps)
        sub_colors_after[rows, positions] = 1 - old_color
        
        mono_after = (sub_colors_after.max(axis=1) == sub_colors_after.min(axis=1))
        
        return int(mono_after.sum()) - int(mono_before.sum())
    
    def faster_delta(colors, p):
        """Even faster delta: only check the 3 other positions in each AP."""
        old_color = colors[p]
        ap_idxs = pos_ap_indices[p]
        n_aps = len(ap_idxs)
        if n_aps == 0:
            return 0
        
        sub_aps = aps[ap_idxs]  # shape (n_aps, 4)
        positions = pos_ap_positions[p]  # shape (n_aps,)
        
        # Get colors of the 3 OTHER positions in each AP
        # For each AP, create a mask to exclude position p
        # Then check if all 3 remaining are same color
        
        # Build indices for the other 3 positions
        # For AP i, position pos_i is p, so others are the 3 columns != pos_i
        # We can use: for each row, take all columns except positions[i]
        
        # Alternative: check if the 3 other positions are all 0 or all 1
        # If they are all 0 and old_color was 0 -> was mono, now not mono (delta -1)
        # If they are all 0 and old_color was 1 -> was not mono, now mono (delta +1)
        # If they are all 1 and old_color was 1 -> was mono, now not mono (delta -1)
        # If they are all 1 and old_color was 0 -> was not mono, now mono (delta +1)
        # Otherwise: delta = 0
        
        # Get the 3 other positions' colors
        # Create a copy and set position p to -1 (sentinel), then check
        sub_colors = colors[sub_aps].copy()  # shape (n_aps, 4)
        rows = np.arange(n_aps)
        sub_colors[rows, positions] = -1  # sentinel
        
        # For each row, check if all non-sentinel values are 0 or all are 1
        # Remove sentinel: for each row, the 3 non-sentinel values
        # We can use: count how many are 0 and how many are 1 among non-sentinel
        
        # Create mask of non-sentinel positions
        # Actually, let's just compute: for each AP, what are the 3 other colors?
        
        # Simpler: for each AP, get the 3 other positions
        # other_pos_1, other_pos_2, other_pos_3 for each AP
        
        # Build arrays of the other 3 positions
        other_cols = []
        for pos_val in positions:
            for col in range(4):
                if col != pos_val:
                    other_cols.append(col)
        
        other_cols = np.array(other_cols, dtype=np.int32)  # shape (3*n_aps,)
        
        # Reshape sub_aps to get the other positions
        # For AP i, the other positions are at columns other_cols[3*i:3*i+3]
        # This is tricky with numpy. Let me use a different approach.
        
        # Just use the original method but optimize the copy
        sub_colors_orig = colors[sub_aps]  # shape (n_aps, 4), no copy
        
        # mono_before
        mono_before = (sub_colors_orig.max(axis=1) == sub_colors_orig.min(axis=1))
        
        # For mono_after, we need to check: after flipping p, are all 4 same?
        # This happens iff the other 3 are all (1 - old_color)
        # So: for each AP, check if the 3 other positions all have color (1 - old_color)
        
        # Get the 3 other positions' colors
        # For each AP i, position positions[i] is p
        # The other 3 positions are sub_aps[i, [0,1,2,3] except positions[i]]
        
        # Build a 2D array of the 3 other positions
        # Use advanced indexing
        other_idx = []
        for i in range(n_aps):
            pos_val = positions[i]
            for col in range(4):
                if col != pos_val:
                    other_idx.append((i, col))
        
        other_idx = np.array(other_idx, dtype=np.int32)
        other_colors = sub_aps[other_idx[:, 0], other_idx[:, 1]]  # shape (3*n_aps,)
        other_colors = colors[other_colors]  # shape (3*n_aps,)
        other_colors = other_colors.reshape(n_aps, 3)
        
        # Check if all 3 are (1 - old_color)
        target = 1 - old_color
        all_target = np.all(other_colors == target, axis=1)  # shape (n_aps,)
        
        # mono_after = all_target
        mono_after = all_target
        
        return int(mono_after.sum()) - int(mono_before.sum())
    
    best_colors = None
    best_count = float('inf')
    
    def update_best(colors, count):
        nonlocal best_count, best_colors
        if count < best_count:
            best_count = count
            best_colors = colors.copy()
    
    # Strategy: multiple restarts with simulated annealing + local search
    num_restarts = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        # Initialize
        if restart == 0:
            # Alternating pairs (seed pattern)
            colors = np.array([(i // 2) % 2 for i in range(N)], dtype=np.int8)
        elif restart == 1:
            # Random
            colors = np.random.randint(0, 2, size=N).astype(np.int8)
        elif restart == 2:
            # Thicker blocks: 000111000111...
            colors = np.array([(i // 3) % 2 for i in range(N)], dtype=np.int8)
        elif restart == 3:
            # Thicker blocks: 0000111100001111...
            colors = np.array([(i // 4) % 2 for i in range(N)], dtype=np.int8)
        elif restart == 4:
            # Thicker blocks: 00000111110000011111...
            colors = np.array([(i // 5) % 2 for i in range(N)], dtype=np.int8)
        elif restart == 5:
            # Thicker blocks: 000000111111000000111111...
            colors = np.array([(i // 6) % 2 for i in range(N)], dtype=np.int8)
        else:
            if best_colors is not None:
                colors = best_colors.copy()
                # Perturb ~15% of bits
                n_flips = max(5, N // 7)
                flip_positions = np.random.choice(N, size=n_flips, replace=False)
                colors[flip_positions] ^= 1
            else:
                colors = np.random.randint(0, 2, size=N).astype(np.int8)
        
        current_count = count_mono(colors)
        update_best(colors, current_count)
        
        # Simulated annealing
        T = 3.0
        T_min = 0.05
        cooling = 0.9998
        max_sa_steps = 100000
        steps_done = 0
        
        for step in range(max_sa_steps):
            if time.time() > deadline - 5:
                break
            
            # Pick random position
            p = np.random.randint(0, N)
            
            # Compute delta
            delta = faster_delta(colors, p)
            
            if delta <= 0:
                colors[p] ^= 1
                current_count += delta
                update_best(colors, current_count)
            else:
                # Accept with probability exp(-delta/T)
                if T > T_min:
                    if random.random() < np.exp(-delta / T):
                        colors[p] ^= 1
                        current_count += delta
                # else reject (T too low)
            
            T = max(T_min, T * cooling)
            steps_done += 1
        
        # Local search: hill climbing
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for p in range(N):
                if time.time() > deadline - 3:
                    break
                delta = faster_delta(colors, p)
                if delta < 0:
                    colors[p] ^= 1
                    current_count += delta
                    improved = True
                    update_best(colors, current_count)
        
        # Second pass of local search (might find more improvements)
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for p in range(N):
                if time.time() > deadline - 2:
                    break
                delta = faster_delta(colors, p)
                if delta < 0:
                    colors[p] ^= 1
                    current_count += delta
                    improved = True
                    update_best(colors, current_count)
    
    if best_colors is not None:
        return [int(c) for c in best_colors]
    else:
        return [(i // 2) % 2 for i in range(N)]
