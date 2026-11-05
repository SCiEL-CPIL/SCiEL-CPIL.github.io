import math
import time
import random
import numpy as np

def construct():
    N = 256
    deadline = time.time() + 145
    
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
    
    def all_deltas(cols_np):
        """Vectorized: compute delta for all positions in parallel."""
        deltas = np.zeros(N, dtype=np.int32)
        for i in range(N):
            c_i = cols_np[i]
            arr = pos_aps_arr[i]
            if len(arr) == 0:
                continue
            others = arr[:, 1:]
            c_others = cols_np[others]
            all_same = (c_others[:, 0] == c_others[:, 1]) & (c_others[:, 1] == c_others[:, 2])
            n_same = int(np.sum(all_same))
            if n_same == 0:
                continue
            matching = int(np.sum(c_others[all_same, 0] == c_i))
            deltas[i] = -(matching) + (n_same - matching)
        return deltas
    
    def batch_polish(cols, current_cost, time_limit):
        """Batch best-improvement polish using vectorized deltas."""
        cols_np = np.array(cols, dtype=np.int8)
        end_time = time.time() + time_limit
        improved = True
        while improved and time.time() < end_time:
            improved = False
            deltas = all_deltas(cols_np)
            neg_deltas = deltas[deltas < 0]
            if len(neg_deltas) > 0:
                # Flip all positions with negative delta simultaneously
                for i in np.where(deltas < 0)[0]:
                    cols[i] = 1 - cols[i]
                    cols_np[i] = 1 - cols_np[i]
                current_cost += int(np.sum(deltas[deltas < 0]))
                improved = True
        return cols, current_cost
    
    def greedy_polish(cols, current_cost):
        """Sequential best-improvement polish."""
        cols_np = np.array(cols, dtype=np.int8)
        improved = True
        while improved:
            improved = False
            for i in range(N):
                d = delta_for_flip(cols_np, i)
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
        for shift in range(1, 10):
            colors = [0] * N
            for i in range(N):
                n = (i + 1 + shift) % (N + 1)
                if n == 0:
                    n = N + 1
                colors[i] = bin(n).count('1') % 2
            patterns.append(colors)
        tm = thue_morse_coloring()
        patterns.append([1 - c for c in tm])
        for block in [2, 3, 4, 5, 7, 8, 13, 16]:
            patterns.append([(i // block) % 2 for i in range(N)])
        patterns.append([i % 2 for i in range(N)])
        patterns.append([(i & 1) ^ ((i >> 1) & 1) for i in range(N)])
        patterns.append([(i & 1) ^ ((i >> 2) & 1) for i in range(N)])
        for seed in range(3):
            rng = random.Random(seed)
            patterns.append([rng.randint(0, 1) for _ in range(N)])
        
        scored = []
        for p in patterns:
            cost = count_mono(np.array(p, dtype=np.int8))
            scored.append((cost, p))
        scored.sort(key=lambda x: x[0])
        return [p for _, p in scored[:8]]
    
    # Get initial patterns
    initial_patterns = generate_structured_patterns()
    
    global_best_colors = initial_patterns[0][:]
    global_best_cost = count_mono(np.array(global_best_colors, dtype=np.int8))
    
    def simulated_annealing(initial_colors, T_start, cooling, min_T, time_limit, seed):
        rng = random.Random(seed)
        colors = initial_colors[:]
        cols_np = np.array(colors, dtype=np.int8)
        current_cost = count_mono(cols_np)
        best_cost = current_cost
        best_colors = colors[:]
        
        T = T_start
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            # 80% single flip, 20% double swap
            if rng.random() < 0.8:
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
                    if rng.random() < 0.5:
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
            else:
                # Double swap: pick two distinct positions
                i = rng.randint(0, N - 1)
                j = rng.randint(0, N - 1)
                while j == i:
                    j = rng.randint(0, N - 1)
                
                if colors[i] == colors[j]:
                    continue
                
                # Compute delta for simultaneous flip of i and j
                # Approximation: sum of individual deltas (ignoring interactions)
                delta_i = delta_for_flip(cols_np, i)
                delta_j = delta_for_flip(cols_np, j)
                # More accurate: compute combined delta
                c_i = cols_np[i]
                c_j = cols_np[j]
                arr_i = pos_aps_arr[i]
                arr_j = pos_aps_arr[j]
                
                combined_delta = 0
                # For each AP containing i
                if len(arr_i) > 0:
                    for ap_idx, o1, o2, o3 in arr_i:
                        others = [cols_np[aps_np[ap_idx, k]] for k in range(4) if aps_np[ap_idx, k] != i]
                        # If AP also contains j, handle separately
                        ap_positions = set(aps_np[ap_idx].tolist())
                        if j in ap_positions:
                            continue
                        c_other_set = set(others)
                        if len(c_other_set) == 1:
                            if others[0] == c_i:
                                combined_delta -= 1
                            else:
                                combined_delta += 1
                # For each AP containing j
                if len(arr_j) > 0:
                    for ap_idx, o1, o2, o3 in arr_j:
                        ap_positions = set(aps_np[ap_idx].tolist())
                        if i in ap_positions:
                            continue
                        others = [cols_np[aps_np[ap_idx, k]] for k in range(4) if aps_np[ap_idx, k] != j]
                        c_other_set = set(others)
                        if len(c_other_set) == 1:
                            if others[0] == c_j:
                                combined_delta -= 1
                            else:
                                combined_delta += 1
                
                if combined_delta < 0:
                    colors[i] = 1 - colors[i]
                    colors[j] = 1 - colors[j]
                    cols_np[i] = 1 - cols_np[i]
                    cols_np[j] = 1 - cols_np[j]
                    current_cost += combined_delta
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_colors = colors[:]
                elif combined_delta == 0:
                    if rng.random() < 0.3:
                        colors[i] = 1 - colors[i]
                        colors[j] = 1 - colors[j]
                        cols_np[i] = 1 - cols_np[i]
                        cols_np[j] = 1 - cols_np[j]
                else:
                    if T > min_T:
                        prob = math.exp(-combined_delta / T)
                        if rng.random() < prob:
                            colors[i] = 1 - colors[i]
                            colors[j] = 1 - colors[j]
                            cols_np[i] = 1 - cols_np[i]
                            cols_np[j] = 1 - cols_np[j]
                            current_cost += combined_delta
                            if current_cost < best_cost:
                                best_cost = current_cost
                                best_colors = colors[:]
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        return best_colors, best_cost
    
    # Multiple restarts
    num_restarts = 20
    time_per_restart = 135.0 / num_restarts
    
    for restart in range(num_restarts):
        if time.time() > deadline - 8:
            break
        
        # Choose initial
        if restart < len(initial_patterns):
            initial = initial_patterns[restart % len(initial_patterns)][:]
        elif restart < 12:
            initial = global_best_colors[:]
            n_flips = random.randint(N // 5, N // 3)
            for _ in range(n_flips):
                idx = random.randint(0, N - 1)
                initial[idx] = 1 - initial[idx]
        else:
            initial = global_best_colors[:]
            n_flips = random.randint(N // 10, N // 5)
            for _ in range(n_flips):
                idx = random.randint(0, N - 1)
                initial[idx] = 1 - initial[idx]
        
        T_start = 1.5 + 0.4 * (restart % 5)
        cooling = 0.99995
        min_T = 0.005
        seed = 42 + restart * 131
        
        sa_colors, sa_cost = simulated_annealing(
            initial, T_start, cooling, min_T, time_per_restart * 0.65, seed
        )
        
        # Batch polish first (fast), then sequential polish
        batch_colors, batch_cost = batch_polish(sa_colors, sa_cost, 2.0)
        polished_colors, polished_cost = greedy_polish(batch_colors, batch_cost)
        
        if polished_cost < global_best_cost:
            global_best_cost = polished_cost
            global_best_colors = polished_colors[:]
    
    # Final polish
    if time.time() < deadline - 3:
        final_colors, final_cost = batch_polish(global_best_colors[:], global_best_cost, 3.0)
        if final_cost < global_best_cost:
            global_best_cost = final_cost
            global_best_colors = final_colors
        final_colors2, final_cost2 = greedy_polish(global_best_colors[:], global_best_cost)
        if final_cost2 < global_best_cost:
            global_best_cost = final_cost2
            global_best_colors = final_colors2
    
    return global_best_colors
