"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s/150s kill

    n = 256
    # Generate all 4-term arithmetic progressions
    # (a, a+d, a+2d, a+3d)
    aps = []
    ap_indices_by_pos = [[] for _ in range(n)]
    
    # 1-indexed logic for generation, 0-indexed for storage
    for d in range(1, 85):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            
            ap = (i0, i1, i2, i3)
            ap_idx = len(aps)
            aps.append(ap)
            
            ap_indices_by_pos[i0].append(ap_idx)
            ap_indices_by_pos[i1].append(ap_idx)
            ap_indices_by_pos[i2].append(ap_idx)
            ap_indices_by_pos[i3].append(ap_idx)
            
    total_aps = len(aps)
    
    # Precompute numpy arrays for fast delta computation
    aps_arr = np.array(aps, dtype=np.int32)  # shape (M, 4)
    pos_aps_arr = [np.array(lst, dtype=np.int32) for lst in ap_indices_by_pos]
    
    # Precompute column indices for each position in each AP it belongs to
    pos_col_arr = []
    for k in range(n):
        if len(pos_aps_arr[k]) == 0:
            pos_col_arr.append(np.array([], dtype=np.int32))
        else:
            affected = aps_arr[pos_aps_arr[k]]
            cols = np.zeros(len(pos_aps_arr[k]), dtype=np.int32)
            for i in range(len(pos_aps_arr[k])):
                row = affected[i]
                for j in range(4):
                    if row[j] == k:
                        cols[i] = j
                        break
            pos_col_arr.append(cols)

    def count_mono(colors):
        """Count monochromatic 4-APs using numpy."""
        col = np.array(colors, dtype=np.int8)
        a0 = col[aps_arr[:, 0]]
        a1 = col[aps_arr[:, 1]]
        a2 = col[aps_arr[:, 2]]
        a3 = col[aps_arr[:, 3]]
        mono = (a0 == a1) & (a1 == a2) & (a2 == a3)
        return int(np.sum(mono))

    def get_delta(k, colors):
        """
        Compute change in mono count if we flip colors[k].
        Uses precomputed column indices for speed.
        k is 0-indexed.
        """
        ap_indices = pos_aps_arr[k]
        if len(ap_indices) == 0:
            return 0
        
        # Get the affected APs
        affected = aps_arr[ap_indices]  # shape (K, 4)
        
        # Current colors at all positions in affected APs
        col = np.array(colors, dtype=np.int8)
        colors_affected = col[affected]  # shape (K, 4)
        
        # Check which are currently mono
        cur_mono = (colors_affected[:, 0] == colors_affected[:, 1]) & \
                   (colors_affected[:, 1] == colors_affected[:, 2]) & \
                   (colors_affected[:, 2] == colors_affected[:, 3])
        cur_mono_count = int(np.sum(cur_mono))
        
        # After flip: flip the specific position k in each AP
        cols = pos_col_arr[k]
        flipped = colors_affected.copy()
        rows = np.arange(len(ap_indices))
        flipped[rows, cols] = 1 - flipped[rows, cols]
        
        new_mono = (flipped[:, 0] == flipped[:, 1]) & \
                   (flipped[:, 1] == flipped[:, 2]) & \
                   (flipped[:, 2] == flipped[:, 3])
        new_mono_count = int(np.sum(new_mono))
        
        return new_mono_count - cur_mono_count

    def thue_morse(n):
        """Generate Thue-Morse sequence of length n (0-indexed)."""
        return [bin(i).count('1') % 2 for i in range(n)]

    def greedy_polish(colors, score):
        """Steepest descent polish."""
        for _ in range(5):
            best_delta = 0
            best_k = -1
            for k in range(n):
                delta = get_delta(k, colors)
                if delta < best_delta:
                    best_delta = delta
                    best_k = k
            if best_k != -1:
                colors[best_k] = 1 - colors[best_k]
                score += best_delta
            else:
                break
        return colors, score

    # Use Thue-Morse as initial coloring
    best_colors = thue_morse(n)
    best_score = count_mono(best_colors)
    
    # Apply initial greedy polish
    best_colors, best_score = greedy_polish(best_colors, best_score)
    
    # Iterated Local Search (ILS) with Simulated Annealing
    max_restarts = 50
    current_colors = best_colors[:]
    current_score = best_score
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            break
            
        # Perturb the best solution
        num_perturb = random.randint(5, 15)
        current_colors = best_colors[:]
        for _ in range(num_perturb):
            k = random.randint(0, n - 1)
            current_colors[k] = 1 - current_colors[k]
        current_score = count_mono(current_colors)
        
        # Local Search: Simulated Annealing
        T = 1.0
        cooling = 0.9999
        min_T = 0.01
        steps = 0
        max_steps = 20000
        
        while time.time() < deadline and steps < max_steps:
            k = random.randint(0, n - 1)
            delta = get_delta(k, current_colors)
            
            accept = False
            if delta < 0:
                accept = True
            else:
                if delta == 0:
                    accept = True
                else:
                    if T > min_T:
                        prob = math.exp(-delta / T)
                        if random.random() < prob:
                            accept = True
            
            if accept:
                current_colors[k] = 1 - current_colors[k]
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_colors = current_colors[:]
            
            T *= cooling
            steps += 1

        # After SA, do a final greedy polish on the current local optimum
        current_colors, current_score = greedy_polish(current_colors, current_score)
        if current_score < best_score:
            best_score = current_score
            best_colors = current_colors[:]

    return best_colors
# EVOLVE-BLOCK-END
