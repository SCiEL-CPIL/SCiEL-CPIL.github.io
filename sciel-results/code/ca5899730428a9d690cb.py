import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    # Precompute all 4-term arithmetic progressions
    ap_list = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            ap_list.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))

    total_aps = len(ap_list)
    aps = np.array(ap_list, dtype=np.int32)

    # Precompute for each position, the indices of the other 3 elements in every AP containing it.
    pos_ap_others = [None] * N
    for p in range(N):
        others_list = []
        for i in range(total_aps):
            row = aps[i]
            if p in row:
                idx = np.where(row == p)[0][0]
                others = [row[j] for j in range(4) if j != idx]
                others_list.append(others)
        if others_list:
            pos_ap_others[p] = np.array(others_list, dtype=np.int32)
        else:
            pos_ap_others[p] = np.empty((0, 3), dtype=np.int32)

    def count_mono(colors):
        """Count monochromatic 4-APs using numpy. colors is numpy array of 0/1."""
        c = colors[aps]  # shape (total_aps, 4)
        mono = (c.max(axis=1) == c.min(axis=1))
        return int(mono.sum())

    def fast_delta(colors, p):
        """Compute change in mono count if we flip position p using precomputed others."""
        old_color = colors[p]
        others = pos_ap_others[p]
        n_aps = others.shape[0]
        if n_aps == 0:
            return 0

        # Get colors of the other 3 positions in each AP
        other_colors = colors[others]  # shape (n_aps, 3)

        # Check if the 3 other positions are all the same color
        all_same = (other_colors[:, 0] == other_colors[:, 1]) & (other_colors[:, 1] == other_colors[:, 2])

        # Before flip: AP is mono if all 3 others are same AND that color is old_color
        mono_before = all_same & (other_colors[:, 0] == old_color)

        # After flip: AP is mono if all 3 others are same AND that color is new_color
        new_color = 1 - old_color
        mono_after = all_same & (other_colors[:, 0] == new_color)

        return int(mono_after.sum()) - int(mono_before.sum())

    def fast_delta_multi(colors, flip_mask):
        """Compute change in mono count if we flip multiple positions at once.
        flip_mask is a boolean array of length N."""
        flipped = np.where(flip_mask)[0]
        if len(flipped) == 0:
            return 0
        
        old_count = count_mono(colors)
        new_colors = colors.copy()
        new_colors[flip_mask] ^= 1
        new_count = count_mono(new_colors)
        return new_count - old_count

    def local_search(colors, max_passes=3, allow_multi=False):
        """Perform hill-climbing local search. Returns (colors, count)."""
        count = count_mono(colors)
        for _ in range(max_passes):
            if time.time() > deadline - 3:
                break
            improved = False
            
            # Standard single-bit flips
            for p in range(N):
                if time.time() > deadline - 3:
                    break
                delta = fast_delta(colors, p)
                if delta < 0:
                    colors[p] ^= 1
                    count += delta
                    improved = True
            
            # Multi-bit flips (2-3 bits) to escape barriers
            if allow_multi:
                for _ in range(20):
                    if time.time() > deadline - 3:
                        break
                    # Try flipping 2 random bits
                    p1, p2 = np.random.randint(0, N, 2)
                    if p1 == p2:
                        continue
                    flip_mask = np.zeros(N, dtype=bool)
                    flip_mask[p1] = True
                    flip_mask[p2] = True
                    delta = fast_delta_multi(colors, flip_mask)
                    if delta < 0:
                        colors[flip_mask] ^= 1
                        count += delta
                        improved = True

            if not improved:
                break
        return colors, count

    def simulated_annealing(colors, max_steps=5000, allow_multi=False):
        """Apply simulated annealing to escape local optima."""
        count = count_mono(colors)
        best_colors = colors.copy()
        best_count = count
        T = 5.0
        cooling = 0.9995
        min_temp = 0.1

        for step in range(max_steps):
            if time.time() > deadline - 10:
                break

            if allow_multi and random.random() < 0.1:
                # Multi-bit flip
                num_flips = random.choice([2, 3])
                flip_idx = np.random.choice(N, num_flips, replace=False)
                flip_mask = np.zeros(N, dtype=bool)
                flip_mask[flip_idx] = True
                delta = fast_delta_multi(colors, flip_mask)
                if delta <= 0:
                    colors[flip_mask] ^= 1
                    count += delta
                    if count < best_count:
                        best_count = count
                        best_colors = colors.copy()
                else:
                    if T > min_temp and random.random() < np.exp(-delta / T):
                        colors[flip_mask] ^= 1
                        count += delta
            else:
                # Single-bit flip
                p = np.random.randint(0, N)
                delta = fast_delta(colors, p)

                if delta <= 0:
                    colors[p] ^= 1
                    count += delta
                    if count < best_count:
                        best_count = count
                        best_colors = colors.copy()
                else:
                    if T > min_temp and random.random() < np.exp(-delta / T):
                        colors[p] ^= 1
                        count += delta

            T = max(min_temp, T * cooling)

        return best_colors, best_count

    best_colors = None
    best_count = float('inf')

    def update_best(colors, count):
        nonlocal best_count, best_colors
        if count < best_count:
            best_count = count
            best_colors = colors.copy()

    # Initialize with diverse starting points
    init_patterns = []

    # Pattern 1: Alternating pairs
    init_patterns.append(np.array([(i // 2) % 2 for i in range(N)], dtype=np.int8))

    # Pattern 2: Blocks of 3
    init_patterns.append(np.array([(i // 3) % 2 for i in range(N)], dtype=np.int8))

    # Pattern 3: Blocks of 4
    init_patterns.append(np.array([(i // 4) % 2 for i in range(N)], dtype=np.int8))

    # Pattern 4: Rademacher-like (based on binary representation)
    pat = np.zeros(N, dtype=np.int8)
    for i in range(N):
        # Use the second-highest bit to create a pattern
        pat[i] = (i >> 4) % 2
    init_patterns.append(pat)

    # Pattern 5: Product of two low-frequency patterns
    pat1 = np.array([(i // 8) % 2 for i in range(N)], dtype=np.int8)
    pat2 = np.array([(i >> 3) % 2 for i in range(N)], dtype=np.int8)
    init_patterns.append((pat1 * pat2).astype(np.int8))

    # Add several random initializations
    for _ in range(8):
        init_patterns.append(np.random.randint(0, 2, size=N).astype(np.int8))

    # Process each initialization
    for init in init_patterns:
        if time.time() > deadline - 20:
            break

        colors = init.copy()
        # Local search with multi-bit flips
        colors, cnt = local_search(colors, max_passes=3, allow_multi=True)
        update_best(colors, cnt)

        # Simulated annealing to escape local optima
        if time.time() > deadline - 20:
            break
        sa_colors, sa_cnt = simulated_annealing(colors.copy(), max_steps=3000, allow_multi=True)
        update_best(sa_colors, sa_cnt)

        # Final local search on SA result
        if time.time() > deadline - 5:
            break
        fs_colors, fs_cnt = local_search(sa_colors.copy(), max_passes=3, allow_multi=True)
        update_best(fs_colors, fs_cnt)

    # Additional restarts with random perturbations of best solution
    if best_colors is not None and time.time() < deadline - 30:
        for restart in range(10):
            if time.time() > deadline - 15:
                break

            # Start from best, perturb some bits
            colors = best_colors.copy()
            # Flip ~10% of bits
            flip_mask = (np.random.rand(N) < 0.10).astype(np.int8)
            colors[flip_mask] ^= 1

            colors, cnt = local_search(colors, max_passes=2, allow_multi=True)
            update_best(colors, cnt)

            if time.time() > deadline - 15:
                break
            sa_colors, sa_cnt = simulated_annealing(colors.copy(), max_steps=2000, allow_multi=True)
            update_best(sa_colors, sa_cnt)

            if time.time() > deadline - 5:
                break
            fs_colors, fs_cnt = local_search(sa_colors.copy(), max_passes=2, allow_multi=True)
            update_best(fs_colors, fs_cnt)

    # Final polish on best
    if best_colors is not None:
        if time.time() < deadline - 3:
            final_colors, final_count = local_search(best_colors.copy(), max_passes=5, allow_multi=True)
            update_best(final_colors, final_count)

    if best_colors is not None:
        return [int(c) for c in best_colors]
    else:
        return [(i // 2) % 2 for i in range(N)]
