"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def construct():
    deadline = time.time() + 110  # Safety margin before 120s limit
    random.seed(42)
    np.random.seed(42)
    
    N = 256
    # Precompute all 4-term APs as index tuples (0-indexed)
    aps = []
    for d in range(1, 86):  # d from 1 to 85 (since 3*d <= 255)
        max_start = N - 3 * d
        for a in range(1, max_start + 1):
            i0 = a - 1
            i1 = i0 + d
            i2 = i0 + 2 * d
            i3 = i0 + 3 * d
            aps.append((i0, i1, i2, i3))
    
    n_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)
    
    # Precompute for each position: the list of AP indices it appears in
    pos_ap_indices = [[] for _ in range(N)]
    for idx in range(n_aps):
        i0, i1, i2, i3 = aps_arr[idx]
        pos_ap_indices[i0].append(idx)
        pos_ap_indices[i1].append(idx)
        pos_ap_indices[i2].append(idx)
        pos_ap_indices[i3].append(idx)
    
    pos_ap_indices_arr = [np.array(lst, dtype=np.int32) if lst else np.array([], dtype=np.int32) for lst in pos_ap_indices]
    
    def count_mono(colors):
        """Count monochromatic 4-APs given colors array of 0/1."""
        # Vectorized count
        sub_colors = colors[aps_arr]  # shape (n_aps, 4)
        mono = np.all(sub_colors == sub_colors[:, 0:1], axis=1)
        return int(np.sum(mono))
    
    def calc_delta(pos, colors):
        """Calculate change in mono count if we flip position pos (vectorized)."""
        ap_idxs = pos_ap_indices_arr[pos]
        if len(ap_idxs) == 0:
            return 0
        sub_aps = aps_arr[ap_idxs]  # shape (m, 4)
        sub_colors = colors[sub_aps]  # shape (m, 4)
        pos_color = colors[pos]
        matches = (sub_colors == pos_color).sum(axis=1)  # shape (m,)
        delta = -np.sum(matches == 4) + np.sum(matches == 1)
        return int(delta)
    
    def calc_all_deltas(colors):
        """Calculate delta for all 256 positions at once."""
        deltas = np.zeros(N, dtype=np.int32)
        for pos in range(N):
            deltas[pos] = calc_delta(pos, colors)
        return deltas
    
    # Initialize with alternating blocks of 2
    colors = np.array([(i // 2) % 2 for i in range(N)], dtype=np.int8)
    current_cost = count_mono(colors)
    best_colors = colors.copy()
    best_cost = current_cost
    
    # Simulated annealing with batched moves
    n_restarts = 8
    restart_time = (110.0 - 15.0) / n_restarts  # ~12s per restart, leaving 15s for polish
    
    for restart in range(n_restarts):
        if restart > 0:
            # Perturb best solution for new restart
            colors = best_colors.copy()
            # Randomly flip some positions
            n_flips = random.randint(10, 30)
            flip_positions = random.sample(range(N), n_flips)
            for p in flip_positions:
                colors[p] = 1 - colors[p]
            current_cost = count_mono(colors)
        
        t = 2.0
        cooling = 0.998
        min_t = 0.005
        restart_deadline = time.time() + restart_time
        
        while time.time() < restart_deadline:
            if time.time() > restart_deadline - 0.5:
                break
            
            # Batch move: evaluate all 256 positions and flip the best few
            all_deltas = calc_all_deltas(colors)
            
            # Find positions with negative delta (improvements)
            improving = np.where(all_deltas < 0)[0]
            
            if len(improving) > 0:
                # Flip all improving positions (greedy batch)
                for pos in improving:
                    colors[pos] = 1 - colors[pos]
                    current_cost += all_deltas[pos]
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
            else:
                # No single-flip improvement; try random flips with annealing acceptance
                n_batch = random.randint(1, 3)
                batch_positions = random.sample(range(N), n_batch)
                for pos in batch_positions:
                    delta = calc_delta(pos, colors)
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
    
    # Final polish: multiple passes of greedy local search
    polish_deadline = deadline - 3
    for polish_pass in range(20):
        if time.time() > polish_deadline:
            break
        improved = True
        while improved and time.time() < polish_deadline:
            improved = False
            # Single flips - vectorized
            all_deltas = calc_all_deltas(best_colors)
            improving = np.where(all_deltas < 0)[0]
            if len(improving) > 0:
                for pos in improving:
                    best_colors[pos] = 1 - best_colors[pos]
                    best_cost += all_deltas[pos]
                improved = True
            # 2-swap moves
            if time.time() > polish_deadline:
                break
            for pos1 in range(N):
                if time.time() > polish_deadline:
                    break
                for pos2 in range(pos1 + 1, N):
                    if time.time() > polish_deadline:
                        break
                    old_c1 = best_colors[pos1]
                    old_c2 = best_colors[pos2]
                    best_colors[pos1] = 1 - old_c1
                    best_colors[pos2] = 1 - old_c2
                    new_cost = count_mono(best_colors)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        improved = True
                    else:
                        best_colors[pos1] = old_c1
                        best_colors[pos2] = old_c2
    
    return best_colors.tolist()
# EVOLVE-BLOCK-END
