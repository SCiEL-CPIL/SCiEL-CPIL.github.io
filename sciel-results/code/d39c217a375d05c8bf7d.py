import time
import random
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    deadline = start_time + 140.0

    # Precompute all 4-term APs as index arrays
    ap_indices = []
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 3 * d
        for a in range(max_a):
            ap_indices.append((a, a + d, a + 2 * d, a + 3 * d))
    ap_arr = np.array(ap_indices, dtype=np.int32)  # shape (M, 4)
    M = len(ap_arr)

    def count_mono(colors):
        """Count monochromatic 4-APs using vectorized operations."""
        c0 = colors[ap_arr[:, 0]]
        c1 = colors[ap_arr[:, 1]]
        c2 = colors[ap_arr[:, 2]]
        c3 = colors[ap_arr[:, 3]]
        return int(np.sum((c0 == c1) & (c1 == c2) & (c2 == c3)))

    def compute_all_deltas(colors):
        """
        Compute delta for flipping each position 0..n-1.
        Returns array of length n where delta[i] = change in mono count if we flip position i.
        
        For each AP (i0, i1, i2, i3) with colors (c0, c1, c2, c3):
        - old_mono = 1 if all four colors equal
        - If we flip position p in the AP, new_mono = 1 if the other 3 colors are all equal to (1 - c_p)
        
        Vectorized approach:
        For each AP, compute:
          - old_mono: all 4 equal
          - For each of the 4 positions in the AP, the "others all equal flipped_color" condition
        """
        c0 = colors[ap_arr[:, 0]]
        c1 = colors[ap_arr[:, 1]]
        c2 = colors[ap_arr[:, 2]]
        c3 = colors[ap_arr[:, 3]]
        
        old_mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        
        # If we flip position 0 (color c0 -> 1-c0):
        # new_mono = 1 if c1 == c2 == c3 == 1-c0
        new_mono_0 = (c1 == c2) & (c2 == c3) & (c1 == 1 - c0)
        
        # If we flip position 1 (color c1 -> 1-c1):
        # new_mono = 1 if c0 == c2 == c3 == 1-c1
        new_mono_1 = (c0 == c2) & (c2 == c3) & (c0 == 1 - c1)
        
        # If we flip position 2 (color c2 -> 1-c2):
        # new_mono = 1 if c0 == c1 == c3 == 1-c2
        new_mono_2 = (c0 == c1) & (c1 == c3) & (c0 == 1 - c2)
        
        # If we flip position 3 (color c3 -> 1-c3):
        # new_mono = 1 if c0 == c1 == c2 == 1-c3
        new_mono_3 = (c0 == c1) & (c1 == c2) & (c0 == 1 - c3)
        
        # Delta for each AP at each position: new_mono - old_mono
        delta_0 = new_mono_0.astype(np.int32) - old_mono.astype(np.int32)
        delta_1 = new_mono_1.astype(np.int32) - old_mono.astype(np.int32)
        delta_2 = new_mono_2.astype(np.int32) - old_mono.astype(np.int32)
        delta_3 = new_mono_3.astype(np.int32) - old_mono.astype(np.int32)
        
        # Accumulate deltas per position
        deltas = np.zeros(n, dtype=np.int32)
        np.add.at(deltas, ap_arr[:, 0], delta_0)
        np.add.at(deltas, ap_arr[:, 1], delta_1)
        np.add.at(deltas, ap_arr[:, 2], delta_2)
        np.add.at(deltas, ap_arr[:, 3], delta_3)
        
        return deltas

    # Generate diverse initializations
    initializers = []
    
    # Thue-Morse
    tm = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    initializers.append(tm)
    
    # Random restarts
    for _ in range(10):
        initializers.append(np.random.randint(0, 2, n).astype(np.int8))
    
    # Periodic patterns
    initializers.append(np.array([(i // 2) % 2 for i in range(n)], dtype=np.int8))
    initializers.append(np.array([(i // 3) % 2 for i in range(n)], dtype=np.int8))
    initializers.append(np.array([(i // 4) % 2 for i in range(n)], dtype=np.int8))
    initializers.append(np.array([(i // 5) % 2 for i in range(n)], dtype=np.int8))
    initializers.append(np.array([(i // 8) % 2 for i in range(n)], dtype=np.int8))
    
    # Walsh-Hadamard-like
    initializers.append(np.array([(i & 1) ^ ((i >> 1) & 1) ^ ((i >> 2) & 1) for i in range(n)], dtype=np.int8))
    
    best_score = float('inf')
    best_colors = None
    
    for init in initializers:
        if time.time() > deadline:
            break
        
        colors = init.copy()
        current_score = count_mono(colors)
        
        if current_score < best_score:
            best_score = current_score
            best_colors = colors.copy()
        
        # Fast local search with vectorized deltas
        max_sweeps = 500
        for sweep in range(max_sweeps):
            if time.time() > deadline:
                break
            
            deltas = compute_all_deltas(colors)
            
            # Find all positions where flipping improves
            improving = deltas < 0
            if not np.any(improving):
                break
            
            # Apply all improving flips at once (they're independent in terms of delta computation
            # but we should be careful - applying multiple flips changes the landscape)
            # To be safe, apply in order of most improvement, one at a time, but recompute deltas
            # periodically for efficiency.
            
            # Strategy: apply all improving flips, then recompute
            flip_mask = improving
            colors[flip_mask] = 1 - colors[flip_mask]
            current_score = count_mono(colors)
            
            if current_score < best_score:
                best_score = current_score
                best_colors = colors.copy()
            
            # After a batch flip, check if we can do more
            deltas = compute_all_deltas(colors)
            if not np.any(deltas < 0):
                break
        
        # Simulated annealing-style perturbation to escape local minima
        if time.time() < deadline - 5.0:
            # Perturb: flip k random positions
            k = random.randint(5, 20)
            perturb_colors = best_colors.copy()
            flip_pos = np.random.choice(n, size=k, replace=False)
            perturb_colors[flip_pos] = 1 - perturb_colors[flip_pos]
            perturb_score = count_mono(perturb_colors)
            
            # Local search from perturbed state
            colors = perturb_colors
            current_score = perturb_score
            
            for sweep in range(200):
                if time.time() > deadline:
                    break
                
                deltas = compute_all_deltas(colors)
                improving = deltas < 0
                if not np.any(improving):
                    break
                
                colors[improving] = 1 - colors[improving]
                current_score = count_mono(colors)
                
                if current_score < best_score:
                    best_score = current_score
                    best_colors = colors.copy()
    
    if best_colors is None:
        best_colors = np.array([0] * n, dtype=np.int8)
    
    return best_colors.tolist()
