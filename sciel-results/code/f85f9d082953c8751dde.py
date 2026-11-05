import math
import time
import random
import numpy as np

def construct():
    N = 256
    deadline = time.time() + 135
    
    # Precompute all 4-APs
    aps_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(max_a):
            aps_list.append((a, a+d, a+2*d, a+3*d))
    
    aps_np = np.array(aps_list, dtype=np.int32)  # shape (M, 4)
    M = len(aps_np)
    
    # For each position, precompute which APs it belongs to and the other 3 positions
    pos_aps = [[] for _ in range(N)]
    for idx, (a, b, c, d_pos) in enumerate(aps_list):
        pos_aps[a].append((idx, b, c, d_pos))
        pos_aps[b].append((idx, a, c, d_pos))
        pos_aps[c].append((idx, a, b, d_pos))
        pos_aps[d_pos].append((idx, a, b, c))
    
    # Convert to numpy arrays for fast lookup
    pos_aps_arr = []
    for i in range(N):
        if pos_aps[i]:
            arr = np.array(pos_aps[i], dtype=np.int32)
            pos_aps_arr.append(arr)
        else:
            pos_aps_arr.append(np.empty((0, 4), dtype=np.int32))
    
    def count_mono(cols_np):
        """Vectorized count of monochromatic 4-APs."""
        c0 = cols_np[aps_np[:, 0]]
        c1 = cols_np[aps_np[:, 1]]
        c2 = cols_np[aps_np[:, 2]]
        c3 = cols_np[aps_np[:, 3]]
        return int(np.sum((c0 == c1) & (c1 == c2) & (c2 == c3)))
    
    def delta_for_flip(cols_np, i):
        """Calculate cost delta if we flip position i."""
        c_i = cols_np[i]
        arr = pos_aps_arr[i]
        if len(arr) == 0:
            return 0
        others = arr[:, 1:]  # shape (k, 3)
        c_others = cols_np[others]  # shape (k, 3)
        all_same = (c_others[:, 0] == c_others[:, 1]) & (c_others[:, 1] == c_others[:, 2])
        n_same = int(np.sum(all_same))
        if n_same == 0:
            return 0
        matching = int(np.sum(c_others[all_same, 0] == c_i))
        return -(matching) + (n_same - matching)
    
    def compute_all_deltas(cols_np):
        """Compute flip deltas for all 256 positions in one vectorized pass."""
        # For each AP, determine if it's currently mono and what happens when each of its 4 positions is flipped
        # An AP is mono if all 4 colors are equal.
        # If we flip position j in AP, the AP becomes mono iff the other 3 are all equal AND they differ from the original color of j.
        # Let's think about it differently:
        # For a given AP (a,b,c,d) with colors (ca,cb,cc,cd):
        #   - If all 4 are same color X: flipping any one makes it non-mono (delta -1 for that position from this AP)
        #   - If 3 are X and 1 is Y: flipping the Y one makes it mono (delta +1); flipping an X one makes it non-mono... wait
        #     Actually if 3 are X and 1 is Y, the AP is NOT mono. 
        #     Flipping the Y position to X makes all 4 X -> mono (delta +1 for that position)
        #     Flipping an X position to Y: now we have 2 X and 2 Y -> not mono (delta 0)
        #   - If 2 are X and 2 are Y: not mono. Flipping any position:
        #     If we flip an X to Y: now 1 X, 3 Y -> not mono (delta 0)
        #     If we flip a Y to X: now 3 X, 1 Y -> not mono (delta 0)
        #   - If 1 is X and 3 are Y: not mono. Flipping the X to Y makes all Y -> mono (delta +1)
        #     Flipping a Y to X: now 2 X, 2 Y -> not mono (delta 0)
        
        # So the delta for flipping position i is:
        #   -1 for each AP where all 4 are currently the same color as cols[i] (was mono, becomes non-mono)
        #   +1 for each AP where the other 3 positions are all the same color Y != cols[i] (was non-mono, becomes mono)
        
        # Let's compute this vectorized:
        # For each AP, get the 4 colors
        c0 = cols_np[aps_np[:, 0]]
        c1 = cols_np[aps_np[:, 1]]
        c2 = cols_np[aps_np[:, 2]]
        c3 = cols_np[aps_np[:, 3]]
        
        # All four same
        all4_same = (c0 == c1) & (c1 == c2) & (c2 == c3)
        
        # For each position in the AP, compute the contribution
        # Position 0 (a): 
        #   -1 if all4_same (was mono, flipping makes non-mono)
        #   +1 if c1==c2==c3 and c1 != c0 (other 3 same, different from c0)
        
        deltas = np.zeros(N, dtype=np.int32)
        
        # Contribution from all4_same: -1 for each of the 4 positions
        if np.any(all4_same):
            idxs = np.where(all4_same)[0]
            for pos_col in range(4):
                positions = aps_np[idxs, pos_col]
                np.add.at(deltas, positions, -1)
        
        # Contribution from "other 3 same but different from this position"
        # For position 0: c1==c2==c3 and c1 != c0
        other3_same_diff0 = (c1 == c2) & (c2 == c3) & (c1 != c0)
        if np.any(other3_same_diff0):
            idxs = np.where(other3_same_diff0)[0]
            positions = aps_np[idxs, 0]
            np.add.at(deltas, positions, 1)
        
        # For position 1: c0==c2==c3 and c0 != c1
        other3_same_diff1 = (c0 == c2) & (c2 == c3) & (c0 != c1)
        if np.any(other3_same_diff1):
            idxs = np.where(other3_same_diff1)[0]
            positions = aps_np[idxs, 1]
            np.add.at(deltas, positions, 1)
        
        # For position 2: c0==c1==c3 and c0 != c2
        other3_same_diff2 = (c0 == c1) & (c1 == c3) & (c0 != c2)
        if np.any(other3_same_diff2):
            idxs = np.where(other3_same_diff2)[0]
            positions = aps_np[idxs, 2]
            np.add.at(deltas, positions, 1)
        
        # For position 3: c0==c1==c2 and c0 != c3
        other3_same_diff3 = (c0 == c1) & (c1 == c2) & (c0 != c3)
        if np.any(other3_same_diff3):
            idxs = np.where(other3_same_diff3)[0]
            positions = aps_np[idxs, 3]
            np.add.at(deltas, positions, 1)
        
        return deltas
    
    def greedy_polish(cols, current_cost):
        """Iteratively flip positions that reduce cost, using vectorized delta computation."""
        cols_np = np.array(cols, dtype=np.int8)
        improved = True
        max_iter = 100
        for _ in range(max_iter):
            if time.time() > deadline - 5:
                break
            improved = False
            deltas = compute_all_deltas(cols_np)
            # Find positions with negative delta
            neg_positions = np.where(deltas < 0)[0]
            if len(neg_positions) == 0:
                break
            # Apply all beneficial flips at once (they may interact, but this is a heuristic)
            for i in neg_positions:
                cols[i] = 1 - cols[i]
                cols_np[i] = 1 - cols_np[i]
                current_cost += deltas[i]
            improved = True
            if not improved:
                break
        # Second pass: single flips
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            deltas = compute_all_deltas(cols_np)
            neg_positions = np.where(deltas < 0)[0]
            if len(neg_positions) == 0:
                break
            # Apply one at a time, the most negative first
            neg_positions = neg_positions[np.argsort(deltas[neg_positions])]
            for i in neg_positions:
                d = deltas[i]
                if d < 0:
                    cols[i] = 1 - cols[i]
                    cols_np[i] = 1 - cols_np[i]
                    current_cost += d
                    improved = True
        return cols, current_cost
    
    def thue_morse_coloring():
        colors = [0] * N
        for i in range(N):
            n = i + 1
            colors[i] = bin(n).count('1') % 2
        return colors
    
    def generate_structured_patterns():
        patterns = []
        patterns.append(thue_morse_coloring())
        
        for shift in range(1, 8):
            colors = [0] * N
            for i in range(N):
                n = (i + 1 + shift) % (N + 1)
                if n == 0:
                    n = N + 1
                colors[i] = bin(n).count('1') % 2
            patterns.append(colors)
        
        tm = thue_morse_coloring()
        patterns.append([1 - c for c in tm])
        
        for block in [2, 4, 8, 16]:
            p = [(i // block) % 2 for i in range(N)]
            patterns.append(p)
        
        patterns.append([i % 2 for i in range(N)])
        patterns.append([(i & 1) ^ ((i >> 1) & 1) for i in range(N)])
        patterns.append([(i & 1) ^ ((i >> 2) & 1) for i in range(N)])
        
        rng_init = random.Random(12345)
        for _ in range(4):
            patterns.append([rng_init.randint(0, 1) for _ in range(N)])
        
        scored = []
        for p in patterns:
            cost = count_mono(np.array(p, dtype=np.int8))
            scored.append((cost, p))
        scored.sort(key=lambda x: x[0])
        return [p for _, p in scored[:6]]
    
    initial_patterns = generate_structured_patterns()
    
    global_best_colors = initial_patterns[0][:]
    global_best_cost = count_mono(np.array(global_best_colors, dtype=np.int8))
    
    def simulated_annealing(initial_colors, T_start, cooling, min_T, time_limit, seed=None):
        if seed is not None:
            rng = random.Random(seed)
        else:
            rng = random
        
        colors = initial_colors[:]
        cols_np = np.array(colors, dtype=np.int8)
        current_cost = count_mono(cols_np)
        best_cost = current_cost
        best_colors = colors[:]
        
        T = T_start
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            i = rng.randint(0, N - 1)
            delta = delta_for_flip(cols_np, i)
            
            if delta < 0:
                colors[i] = 1 - colors[i]
                cols_np[i] = 1 - cols_np[i]
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
            elif delta == 0:
                if rng.random() < 0.3:
                    colors[i] = 1 - colors[i]
                    cols_np[i] = 1 - cols_np[i]
            else:
                if T > min_T:
                    prob = math.exp(-delta / T)
                    if rng.random() < prob:
                        colors[i] = 1 - colors[i]
                        cols_np[i] = 1 - cols_np[i]
                        current_cost += delta
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_colors = colors[:]
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        return best_colors, best_cost
    
    # Multiple restarts with varying strategies
    num_restarts = 15
    time_per_restart = 130.0 / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        if restart < len(initial_patterns):
            initial = initial_patterns[restart % len(initial_patterns)][:]
        elif restart < 8:
            initial = global_best_colors[:]
            n_flips = random.randint(N // 4, N // 2)
            for _ in range(n_flips):
                idx = random.randint(0, N - 1)
                initial[idx] = 1 - initial[idx]
        else:
            initial = global_best_colors[:]
            n_flips = random.randint(N // 8, N // 4)
            for _ in range(n_flips):
                idx = random.randint(0, N - 1)
                initial[idx] = 1 - initial[idx]
        
        T_start = 1.5 + 0.5 * (restart % 4)
        cooling = 0.99995
        min_T = 0.003
        seed = 42 + restart * 137
        
        sa_colors, sa_cost = simulated_annealing(
            initial, T_start, cooling, min_T, time_per_restart * 0.7, seed=seed
        )
        
        polished_colors, polished_cost = greedy_polish(sa_colors, sa_cost)
        
        if polished_cost < global_best_cost:
            global_best_cost = polished_cost
            global_best_colors = polished_colors[:]
    
    # Final polish
    if time.time() < deadline - 5:
        final_colors, final_cost = greedy_polish(global_best_colors, global_best_cost)
        if final_cost < global_best_cost:
            global_best_cost = final_cost
            global_best_colors = final_colors
    
    return global_best_colors
