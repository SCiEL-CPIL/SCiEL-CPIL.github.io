"""Local search over 2-colorings of [1..256] to minimize monochromatic 4-APs."""
import time
import random
import numpy as np


def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin before 120s limit

    # Precompute all 4-term APs as index tuples (0-based)
    aps = []
    for a in range(1, N + 1):
        for d in range(1, N + 1):
            if a + 3 * d <= N:
                aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))

    num_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)  # shape (num_aps, 4)

    # Precompute for each position, the AP indices and the column index within that AP
    pos_to_aps_indices = [None] * N
    pos_to_aps_cols = [None] * N
    
    for pos in range(N):
        indices = []
        cols = []
        for ap_idx in range(num_aps):
            ap = aps_arr[ap_idx]
            for col_idx in range(4):
                if ap[col_idx] == pos:
                    indices.append(ap_idx)
                    cols.append(col_idx)
                    break
        if indices:
            pos_to_aps_indices[pos] = np.array(indices, dtype=np.int32)
            pos_to_aps_cols[pos] = np.array(cols, dtype=np.int32)
        else:
            pos_to_aps_indices[pos] = np.array([], dtype=np.int32)
            pos_to_aps_cols[pos] = np.array([], dtype=np.int32)

    def count_mono(colors):
        """Count monochromatic 4-APs. colors is a numpy array of 0/1."""
        c = colors[aps_arr]  # shape (num_aps, 4)
        mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        return int(mono.sum())

    def thue_morse(n):
        """Thue-Morse sequence: parity of number of 1-bits."""
        return bin(n).count('1') % 2

    # Initial coloring: Thue-Morse
    colors = np.array([thue_morse(i) for i in range(N)], dtype=np.int8)
    best_count = count_mono(colors)
    best_colors = colors.copy()

    def delta_for_flip(colors, pos):
        """Compute change in mono count if we flip position `pos`."""
        ap_indices = pos_to_aps_indices[pos]
        if len(ap_indices) == 0:
            return 0
        
        cols = pos_to_aps_cols[pos]
        
        # Get colors for all 4 positions of these APs
        c = colors[aps_arr[ap_indices]]  # shape (len, 4)
        
        # Current mono status
        currently_mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        
        # Flip the specific column for each AP
        # We need to check if flipping makes it mono
        # Instead of copying the whole array, we can compute the new mono status logically
        # Or just use numpy indexing to flip a copy
        c_flipped = c.copy()
        # Flip the bits at the specific columns
        c_flipped[np.arange(len(ap_indices)), cols] ^= 1
        
        after_mono = (c_flipped[:, 0] == c_flipped[:, 1]) & (c_flipped[:, 1] == c_flipped[:, 2]) & (c_flipped[:, 2] == c_flipped[:, 3])
        
        return int(after_mono.sum()) - int(currently_mono.sum())

    # Greedy improvement loop
    improved = True
    while improved and time.time() < deadline:
        improved = False
        best_delta = 0
        best_pos = -1
        # Evaluate all positions
        for pos in range(N):
            d = delta_for_flip(colors, pos)
            if d < best_delta:
                best_delta = d
                best_pos = pos
        if best_pos >= 0 and best_delta < 0:
            colors[best_pos] ^= 1
            best_count += best_delta
            best_colors = colors.copy()
            improved = True

    # If we have time, try a few random restarts with perturbation
    if time.time() < deadline - 10:
        rng = random.Random(42)
        for restart in range(50): # Increased restarts due to speedup
            if time.time() > deadline - 5:
                break
            # Start from best_colors with some random flips
            colors = best_colors.copy()
            num_flips = rng.randint(5, 20)
            flip_positions = rng.sample(range(N), num_flips)
            for p in flip_positions:
                colors[p] ^= 1
            cur_count = count_mono(colors)
            if cur_count < best_count:
                best_count = cur_count
                best_colors = colors.copy()
            # Local search from here
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                best_delta = 0
                best_pos = -1
                for pos in range(N):
                    d = delta_for_flip(colors, pos)
                    if d < best_delta:
                        best_delta = d
                        best_pos = pos
                if best_pos >= 0 and best_delta < 0:
                    colors[best_pos] ^= 1
                    cur_count += best_delta
                    if cur_count < best_count:
                        best_count = cur_count
                        best_colors = colors.copy()
                    improved = True

    return [int(c) for c in best_colors]
