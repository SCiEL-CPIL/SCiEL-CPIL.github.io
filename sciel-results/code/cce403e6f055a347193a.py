import time
import random
import math
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 140  # Safety margin
    
    # 1. Precompute APs
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            p0 = a - 1
            p1 = a + d - 1
            p2 = a + 2 * d - 1
            p3 = a + 3 * d - 1
            aps.append((p0, p1, p2, p3))
    
    num_aps = len(aps)
    
    # Map position to list of AP indices
    pos_aps = [[] for _ in range(n)]
    for idx in range(num_aps):
        p0, p1, p2, p3 = aps[idx]
        pos_aps[p0].append(idx)
        pos_aps[p1].append(idx)
        pos_aps[p2].append(idx)
        pos_aps[p3].append(idx)
    
    # Convert to numpy for vectorized operations
    aps_arr = np.array(aps, dtype=np.int32)  # shape (num_aps, 4)
    
    # Helper to compute initial mono count (vectorized)
    def count_mono(color):
        c = np.array(color, dtype=np.int8)
        # For each AP, check if all 4 positions have same color
        # c[aps_arr[:,0]], c[aps_arr[:,1]], etc.
        mono = (c[aps_arr[:,0]] == c[aps_arr[:,1]]) & \
               (c[aps_arr[:,1]] == c[aps_arr[:,2]]) & \
               (c[aps_arr[:,2]] == c[aps_arr[:,3]])
        return int(np.sum(mono))
    
    # Helper to compute delta for flipping position i (vectorized)
    def calc_delta(color, i):
        c = np.array(color, dtype=np.int8)
        c_i = c[i]
        new_c_i = 1 - c_i
        
        # Get all APs containing position i
        ap_indices = pos_aps[i]
        if not ap_indices:
            return 0
        
        # Get the APs
        sub_aps = aps_arr[ap_indices]  # shape (k, 4)
        
        # For each AP, check the other 3 positions
        # We need to identify which column corresponds to position i
        # and check the other 3
        
        # Create masks for each position in the AP
        # pos0_mask: where sub_aps[:,0] == i
        # etc.
        
        delta = 0
        for col in range(4):
            mask = (sub_aps[:, col] == i)
            if not np.any(mask):
                continue
            
            # For APs where position i is at column col
            # Get the other 3 positions
            other_cols = [c2 for c2 in range(4) if c2 != col]
            
            # Check if currently mono: all 4 are c_i
            # Since c[i] = c_i, check if other 3 are also c_i
            currently_mono = True
            for oc in other_cols:
                currently_mono &= np.all(sub_aps[mask, oc] != i) or True  # placeholder
            
            # More direct approach:
            # For each AP in mask, check colors of other 3 positions
            for row in range(len(sub_aps)):
                if sub_aps[row, col] != i:
                    continue
                # Get other 3 positions
                others = []
                for c2 in range(4):
                    if c2 != col:
                        others.append(c[sub_aps[row, c2]])
                
                # Check if all others are c_i (currently mono)
                if all(v == c_i for v in others):
                    delta -= 1
                
                # Check if all others are new_c_i (becomes mono)
                if all(v == new_c_i for v in others):
                    delta += 1
        
        return delta
    
    # Optimized delta calculation using precomputed neighbor info
    def calc_delta_fast(color_arr, i):
        """
        color_arr: numpy array of colors
        i: position to flip
        Returns delta (change in mono count)
        """
        c_i = color_arr[i]
        new_c_i = 1 - c_i
        
        delta = 0
        for ap_idx in pos_aps[i]:
            p0, p1, p2, p3 = aps[ap_idx]
            
            # Check if currently mono
            if color_arr[p0] == color_arr[p1] == color_arr[p2] == color_arr[p3]:
                # It's currently mono. Flipping i breaks it if i has color c_i
                # (which it does), so delta -= 1
                delta -= 1
            
            # Check if becomes mono after flip
            # After flip, position i has color new_c_i
            # Check if other 3 all have color new_c_i
            others_match = True
            for p in (p0, p1, p2, p3):
                if p != i and color_arr[p] != new_c_i:
                    others_match = False
                    break
            
            if others_match:
                delta += 1
        
        return delta
    
    # Local Search (Greedy) with vectorized mono detection
    def local_search(color, max_iters=50):
        color_arr = np.array(color, dtype=np.int8)
        current_score = count_mono(color)
        improved = True
        while improved and max_iters > 0:
            improved = False
            max_iters -= 1
            
            # Find positions in mono APs (vectorized)
            c = color_arr
            mono_mask = (c[aps_arr[:,0]] == c[aps_arr[:,1]]) & \
                        (c[aps_arr[:,1]] == c[aps_arr[:,2]]) & \
                        (c[aps_arr[:,2]] == c[aps_arr[:,3]])
            
            if not np.any(mono_mask):
                break
            
            # Get all positions in mono APs
            mono_positions = set()
            for idx in range(num_aps):
                if mono_mask[idx]:
                    mono_positions.update(aps[idx])
            
            if not mono_positions:
                break
                
            best_delta = 0
            best_pos = -1
            for p in mono_positions:
                d = calc_delta_fast(color_arr, p)
                if d < best_delta:
                    best_delta = d
                    best_pos = p
                    
            if best_pos != -1:
                color_arr[best_pos] = 1 - color_arr[best_pos]
                current_score += best_delta
                improved = True
        
        return color_arr.tolist(), current_score
    
    # Simulated Annealing with multi-flip moves
    def simulated_annealing(initial_color, T_start=10.0, T_end=0.1, steps=50000):
        color_arr = np.array(initial_color, dtype=np.int8)
        current_score = count_mono(initial_color)
        best_color = color_arr.copy()
        best_score = current_score
        
        T = T_start
        cooling = (T_end / T_start) ** (1.0 / steps)
        
        for step in range(steps):
            if time.time() > deadline - 5:
                break
                
            # Multi-flip move: flip 1-3 random positions
            num_flips = random.choices([1, 2, 3], weights=[0.6, 0.3, 0.1])[0]
            positions = random.sample(range(n), min(num_flips, n))
            
            # Calculate total delta for flipping all positions
            # This is complex due to interactions, so we do it sequentially
            # but accept/reject as a group
            
            old_colors = [color_arr[p] for p in positions]
            
            # Apply flips
            for p in positions:
                color_arr[p] = 1 - color_arr[p]
            
            # Calculate new score
            new_score = count_mono(color_arr.tolist())
            delta = new_score - current_score
            
            # Acceptance
            if delta < 0:
                accept = True
            else:
                if T > 0.001:
                    accept = random.random() < math.exp(-delta / T)
                else:
                    accept = False
                    
            if not accept:
                # Revert
                for idx, p in enumerate(positions):
                    color_arr[p] = old_colors[idx]
            else:
                current_score = new_score
                if current_score < best_score:
                    best_score = current_score
                    best_color = color_arr.copy()
                    
            T *= cooling
            
        return best_color.tolist(), best_score
    
    # Main logic
    random.seed(42)
    
    # Initialize with Thue-Morse
    def thue_morse(n):
        return [bin(i).count('1') % 2 for i in range(n)]
    
    best_overall_color = None
    best_overall_score = float('inf')
    
    # Start with Thue-Morse
    tm = thue_morse(n)
    tm, score_tm = local_search(tm, max_iters=30)
    best_overall_color = tm[:]
    best_overall_score = score_tm
    
    # Run SA from Thue-Morse with more steps
    sa_color, sa_score = simulated_annealing(tm, T_start=8.0, T_end=0.01, steps=30000)
    if sa_score < best_overall_score:
        best_overall_score = sa_score
        best_overall_color = sa_color[:]
    
    # Run SA from Random with diverse seeds
    for trial in range(5):
        if time.time() > deadline - 15:
            break
        random.seed(42 + trial)
        rand_color = [random.randint(0, 1) for _ in range(n)]
        rand_color, rand_score = local_search(rand_color, max_iters=15)
        
        sa_color, sa_score = simulated_annealing(rand_color, T_start=8.0, T_end=0.01, steps=15000)
        if sa_score < best_overall_score:
            best_overall_score = sa_score
            best_overall_color = sa_color[:]
            
    # Final polish with more iterations
    best_overall_color, best_overall_score = local_search(best_overall_color, max_iters=150)
    
    # Additional SA polish
    if time.time() < deadline - 10:
        sa_color, sa_score = simulated_annealing(best_overall_color, T_start=2.0, T_end=0.005, steps=20000)
        if sa_score < best_overall_score:
            best_overall_score = sa_score
            best_overall_color = sa_color[:]
            best_overall_color, best_overall_score = local_search(best_overall_color, max_iters=50)
    
    return best_overall_color
