"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    random.seed(12345)
    np.random.seed(12345)
    
    N = 256
    # Precompute all 4-term APs as index tuples (0-indexed)
    # A 4-AP is (x, x+d, x+2d, x+3d) with 1 <= x and x+3d <= 256, d >= 1
    # 0-indexed: (x-1, x-1+d, x-1+2d, x-1+3d)
    aps = []
    for d in range(1, 86):  # d from 1 to 85
        max_start = N - 3 * d
        for a in range(1, max_start + 1):
            i0 = a - 1
            i1 = i0 + d
            i2 = i0 + 2 * d
            i3 = i0 + 3 * d
            aps.append((i0, i1, i2, i3))
    
    n_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)  # shape (n_aps, 4)
    
    def count_mono(coloring):
        """Count monochromatic 4-APs. coloring is array of 0/1, shape (N,)"""
        # For each AP, check if all 4 positions have same color
        c = coloring[aps_arr]  # shape (n_aps, 4)
        # Monochromatic: all four equal
        # c[:,0] == c[:,1] == c[:,2] == c[:,3]
        mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        return int(mono.sum())
    
    def calc_delta(coloring, idx):
        """Calculate change in mono count if we flip position idx."""
        # Find all APs that include position idx
        # For efficiency, we can precompute which APs contain each position
        # But for now, let's do it naively but vectorized
        mask = (aps_arr == idx).any(axis=1)
        if not mask.any():
            return 0
        sub_aps = aps_arr[mask]
        sub_c = coloring[sub_aps]
        # Current mono count for these APs
        cur_mono = (sub_c[:, 0] == sub_c[:, 1]) & (sub_c[:, 1] == sub_c[:, 2]) & (sub_c[:, 2] == sub_c[:, 3])
        cur_count = cur_mono.sum()
        # After flip: flip the color at position idx in each AP
        flipped = sub_c.copy()
        # For each AP in sub_aps, find which column is idx and flip it
        for j in range(4):
            flip_mask = sub_aps[:, j] == idx
            flipped[flip_mask, j] = 1 - flipped[flip_mask, j]
        new_mono = (flipped[:, 0] == flipped[:, 1]) & (flipped[:, 1] == flipped[:, 2]) & (flipped[:, 2] == flipped[:, 3])
        new_count = new_mono.sum()
        return int(new_count - cur_count)
    
    best_coloring = None
    best_count = float('inf')
    
    # Try multiple initializations
    initializations = []
    
    # 1. Random
    for _ in range(3):
        initializations.append(np.random.randint(0, 2, N))
    
    # 2. Alternating blocks of various sizes
    for block_size in [1, 2, 3, 4, 5, 8]:
        coloring = np.array([(i // block_size) % 2 for i in range(N)])
        initializations.append(coloring)
    
    # 3. Based on binary representation
    coloring = np.array([(i & 1) for i in range(N)])
    initializations.append(coloring)
    coloring = np.array([(i & 2) // 2 for i in range(N)])
    initializations.append(coloring)
    coloring = np.array([(i & 4) // 4 for i in range(N)])
    initializations.append(coloring)
    
    # 4. Random with bias
    for p in [0.3, 0.7]:
        initializations.append((np.random.random(N) < p).astype(np.int32))
    
    for init in initializations:
        if time.time() > deadline - 10:
            break
        
        coloring = init.copy()
        current_count = count_mono(coloring)
        
        if current_count < best_count:
            best_count = current_count
            best_coloring = coloring.copy()
        
        # Local search: try flipping each position, accept if improves
        improved = True
        while improved and time.time() < deadline - 10:
            improved = False
            # Try all positions in random order
            order = np.random.permutation(N)
            for idx in order:
                if time.time() > deadline - 10:
                    break
                delta = calc_delta(coloring, idx)
                if delta < 0:
                    coloring[idx] = 1 - coloring[idx]
                    current_count += delta
                    improved = True
                    if current_count < best_count:
                        best_count = current_count
                        best_coloring = coloring.copy()
                elif delta == 0:
                    # Accept with small probability to escape local minima
                    if random.random() < 0.1:
                        coloring[idx] = 1 - coloring[idx]
                        improved = True
        
        # After local search, try a few random restarts from current state
        for _ in range(2):
            if time.time() > deadline - 10:
                break
            perturbed = coloring.copy()
            # Flip 5-10 random positions
            n_flips = random.randint(5, 10)
            flip_idx = np.random.choice(N, n_flips, replace=False)
            perturbed[flip_idx] = 1 - perturbed[flip_idx]
            current_count = count_mono(perturbed)
            
            # Local search on perturbed
            improved2 = True
            while improved2 and time.time() < deadline - 10:
                improved2 = False
                order = np.random.permutation(N)
                for idx in order:
                    if time.time() > deadline - 10:
                        break
                    delta = calc_delta(perturbed, idx)
                    if delta < 0:
                        perturbed[idx] = 1 - perturbed[idx]
                        current_count += delta
                        improved2 = True
                        if current_count < best_count:
                            best_count = current_count
                            best_coloring = perturbed.copy()
    
    # Return as list of 0/1
    return best_coloring.tolist()
# EVOLVE-BLOCK-END
