import numpy as np
import time
import random

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) with 1 <= a and a+3d <= N
    # Store as 0-indexed tuples
    aps = []
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            if a + 3 * d <= N:
                aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    num_aps = len(aps)
    ap_array = np.array(aps, dtype=np.int32)  # shape (num_aps, 4)
    
    # Build reverse index: for each position, list of ap indices
    pos_to_aps = [[] for _ in range(N)]
    for idx in range(num_aps):
        for pos in ap_array[idx]:
            pos_to_aps[pos].append(idx)
    
    # Convert to numpy arrays for speed
    pos_to_aps_arr = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in pos_to_aps]
    
    def count_mono_fast(colors):
        """Vectorized count of monochromatic 4-term APs."""
        c0 = colors[ap_array[:, 0]]
        c1 = colors[ap_array[:, 1]]
        c2 = colors[ap_array[:, 2]]
        c3 = colors[ap_array[:, 3]]
        return int(np.sum((c0 == c1) & (c1 == c2) & (c2 == c3)))
    
    def thue_morse(n):
        """Thue-Morse sequence value at position n (0-indexed)."""
        return bin(n).count('1') % 2
    
    # Initial coloring: Thue-Morse
    colors = np.array([thue_morse(i) for i in range(N)], dtype=np.int8)
    
    best_colors = colors.copy()
    best_count = count_mono_fast(best_colors)
    
    rng = random.Random(42)
    
    # Local search with simulated annealing
    # Use multiple restarts
    for restart in range(20):
        if time.time() > deadline - 10.0:
            break
        
        if restart == 0:
            colors = np.array([thue_morse(i) for i in range(N)], dtype=np.int8)
        elif restart == 1:
            colors = np.array([(i // 2) % 2 for i in range(N)], dtype=np.int8)
        elif restart == 2:
            colors = np.array([(i // 4) % 2 for i in range(N)], dtype=np.int8)
        elif restart == 3:
            colors = np.array([(i // 8) % 2 for i in range(N)], dtype=np.int8)
        else:
            colors = np.array([rng.randint(0, 1) for _ in range(N)], dtype=np.int8)
        
        cur_count = count_mono_fast(colors)
        if cur_count < best_count:
            best_count = cur_count
            best_colors = colors.copy()
        
        # Simulated annealing
        T = 2.0
        T_min = 0.05
        cooling = 0.9995
        no_improve = 0
        max_no_improve = 5000
        
        while time.time() < deadline - 5.0:
            pos = rng.randint(0, N - 1)
            old_color = colors[pos]
            new_color = 1 - old_color
            
            # Compute delta
            delta = 0
            ap_indices = pos_to_aps_arr[pos]
            if len(ap_indices) > 0:
                # Get all APs containing pos
                ap_subset = ap_array[ap_indices]
                # For each AP, the other 3 positions
                # We need to check if the other 3 are all equal to old_color or new_color
                
                # Create mask for which position in the AP is pos
                # For each AP, find the other 3 positions
                # This is tricky with numpy... let me do it differently
                
                # For each AP containing pos, get the 3 other positions
                other_cols = []
                for ap_idx in ap_indices:
                    ap = ap_array[ap_idx]
                    others = [p for p in ap if p != pos]
                    other_cols.append(others)
                
                if other_cols:
                    other_arr = np.array(other_cols, dtype=np.int32)  # shape (len(ap_indices), 3)
                    other_colors = colors[other_arr]  # shape (len(ap_indices), 3)
                    
                    # Check if all 3 others equal old_color
                    cur_mono = np.all(other_colors == old_color, axis=1)
                    # Check if all 3 others equal new_color
                    new_mono = np.all(other_colors == new_color, axis=1)
                    
                    # delta = (# new_mono) - (# cur_mono)
                    delta = int(np.sum(new_mono) - np.sum(cur_mono))
            
            colors[pos] = new_color
            
            if delta < 0:
                cur_count += delta
                no_improve = 0
                if cur_count < best_count:
                    best_count = cur_count
                    best_colors = colors.copy()
            else:
                if delta > 0 and rng.random() < np.exp(-delta / T):
                    cur_count += delta
                    no_improve += 1
                else:
                    colors[pos] = old_color
                    no_improve += 1
            
            T *= cooling
            if T < T_min:
                T = T_min
            
            if no_improve > max_no_improve:
                break
    
    return [int(c) for c in best_colors]
