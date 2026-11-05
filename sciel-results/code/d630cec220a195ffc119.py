"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    random.seed(42)
    
    # Precompute all 4-term APs as index tuples (0-indexed)
    # A 4-AP is (x, x+d, x+2d, x+3d) with 1 <= x and x+3d <= 256, d >= 1
    # 0-indexed: (x-1, x-1+d, x-1+2d, x-1+3d)
    aps = []
    for d in range(1, 86):  # d from 1 to 85 (since 3*d <= 255)
        max_start = 256 - 3 * d
        for a in range(1, max_start + 1):
            i0 = a - 1
            i1 = i0 + d
            i2 = i0 + 2 * d
            i3 = i0 + 3 * d
            aps.append((i0, i1, i2, i3))
    
    n_aps = len(aps)
    # Convert to numpy array for fast operations
    aps_arr = np.array(aps, dtype=np.int32)
    
    # For each position, precompute which APs it appears in and at which slot
    # pos_aps[pos] = list of (ap_index, slot) where slot is 0,1,2,3
    pos_aps = [[] for _ in range(256)]
    for idx, (i0, i1, i2, i3) in enumerate(aps):
        pos_aps[i0].append((idx, 0))
        pos_aps[i1].append((idx, 1))
        pos_aps[i2].append((idx, 2))
        pos_aps[i3].append((idx, 3))
    
    def count_mono(colors):
        """Count monochromatic 4-APs given colors array of 0/1."""
        count = 0
        for i in range(n_aps):
            a, b, c, d = aps_arr[i]
            if colors[a] == colors[b] == colors[c] == colors[d]:
                count += 1
        return count
    
    def calc_delta(pos, colors):
        """Calculate change in mono count if we flip position pos."""
        delta = 0
        for ap_idx, slot in pos_aps[pos]:
            a, b, c, d = aps_arr[ap_idx]
            # Count how many of the 4 positions currently have the same color as pos
            same_count = 0
            for p in (a, b, c, d):
                if p != pos and colors[p] == colors[pos]:
                    same_count += 1
            # Currently: if all 4 same, it's mono. After flip, if remaining 3 are same, it becomes mono.
            # Before flip: mono if same_count == 3 (all other 3 match pos)
            # After flip: mono if same_count == 0 (all other 3 match the flipped color)
            if same_count == 3:
                delta -= 1  # Was mono, won't be
            elif same_count == 0:
                delta += 1  # Wasn't mono, will be
        return delta
    
    # Initialize with alternating blocks of 2 (same as seed)
    colors = np.array([(i // 2) % 2 for i in range(256)], dtype=np.int8)
    current_cost = count_mono(colors)
    best_colors = colors.copy()
    best_cost = current_cost
    
    # Simulated annealing
    t = 1.0
    cooling = 0.999
    min_t = 0.01
    
    while time.time() < deadline:
        # Check deadline frequently
        if time.time() > deadline - 5:
            break
            
        # Pick a random position to flip
        pos = random.randint(0, 255)
        
        # Calculate delta
        delta = calc_delta(pos, colors)
        
        # Accept if improvement or with probability exp(-delta/t)
        if delta < 0 or (delta > 0 and random.random() < np.exp(-delta / t)):
            colors[pos] = 1 - colors[pos]
            current_cost += delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
        
        # Cool down
        t *= cooling
        if t < min_t:
            t = min_t
    
    # Final polish: greedy local search
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        for pos in range(256):
            if time.time() > deadline - 2:
                break
            delta = calc_delta(pos, best_colors)
            if delta < 0:
                best_colors[pos] = 1 - best_colors[pos]
                best_cost += delta
                improved = True
    
    return best_colors.tolist()
# EVOLVE-BLOCK-END
