import numpy as np
import time
import random

def construct():
    N = 256
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Precompute all 4-APs as index arrays (0-indexed)
    ap_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(max_a):
            idxs = np.array([a, a + d, a + 2 * d, a + 3 * d])
            ap_list.append(idxs)
    
    if not ap_list:
        return [0] * N
        
    ap_array = np.array(ap_list)  # shape (num_aps, 4)
    num_aps = len(ap_list)
    
    # Precompute for pair-swap delta:
    # For a swap of positions i and j, the delta in cost depends on:
    # - APs containing i but not j: flipping i's color
    # - APs containing j but not i: flipping j's color  
    # - APs containing both i and j: swapping their colors
    # We'll compute this on-the-fly with vectorized ops for speed.
    
    # For each position, list of AP indices containing it
    aps_at = [[] for _ in range(N)]
    for ap_idx in range(num_aps):
        for pos in ap_array[ap_idx]:
            aps_at[pos].append(ap_idx)
    
    def count_mono(cols):
        """Vectorized count of monochromatic 4-APs."""
        # cols: 1D array of 0/1, length N
        # For each AP, check if all 4 positions have same color
        c = cols[ap_array]  # shape (num_aps, 4)
        # monochromatic if all equal: c[:,0]==c[:,1] and c[:,1]==c[:,2] and c[:,2]==c[:,3]
        mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        return int(np.sum(mono))
    
    def delta_for_swap(cols, i, j):
        """
        Compute cost delta if we swap colors at positions i and j.
        Only APs containing i or j are affected.
        """
        if i == j:
            return 0
        
        c_i = cols[i]
        c_j = cols[j]
        
        if c_i == c_j:
            # Swapping same-colored positions: no change
            return 0
        
        # After swap: position i gets c_j, position j gets c_i
        # APs affected: those containing i or j
        affected_aps = set(aps_at[i]) | set(aps_at[j])
        
        delta = 0
        for ap_idx in affected_aps:
            ap = ap_array[ap_idx]
            # Get colors of the 4 positions in this AP
            colors_ap = cols[ap]
            
            # Check if currently monochromatic
            is_mono_now = (colors_ap[0] == colors_ap[1] == colors_ap[2] == colors_ap[3])
            
            # Compute new colors after swap
            new_colors_ap = colors_ap.copy()
            for k in range(4):
                if ap[k] == i:
                    new_colors_ap[k] = c_j
                elif ap[k] == j:
                    new_colors_ap[k] = c_i
            
            is_mono_new = (new_colors_ap[0] == new_colors_ap[1] == new_colors_ap[2] == new_colors_ap[3])
            
            if is_mono_now and not is_mono_new:
                delta -= 1
            elif not is_mono_now and is_mono_new:
                delta += 1
        
        return delta
    
    def pair_swap_polish(colors, current_cost, max_sweeps=50):
        """Apply beneficial pair swaps until no improvement."""
        improved = True
        sweeps = 0
        while improved and sweeps < max_sweeps:
            improved = False
            sweeps += 1
            # Try all pairs (or a random subset for speed)
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                for j in range(i + 1, N):
                    d = delta_for_swap(colors, i, j)
                    if d < 0:
                        colors[i], colors[j] = colors[j], colors[i]
                        current_cost += d
                        improved = True
        return colors, current_cost
    
    def generate_good_initial():
        """Generate a structurally informed initial coloring."""
        # Use a pattern based on modular arithmetic that breaks APs
        # Color by i mod 4: positions 0,1 -> 0, positions 2,3 -> 1
        # This breaks many short APs
        return np.array([(i % 4) >= 2 for i in range(N)], dtype=np.int8)
    
    def simulated_annealing_with_swaps(initial_colors, T_start, cooling, min_T, time_limit):
        """SA using pair swaps as moves."""
        colors = initial_colors.copy()
        current_cost = count_mono(colors)
        best_cost = current_cost
        best_colors = colors.copy()
        
        T = T_start
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            # Pick a random pair
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i == j:
                continue
                
            delta = delta_for_swap(colors, i, j)
            
            if delta < 0:
                colors[i], colors[j] = colors[j], colors[i]
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
            elif delta == 0:
                # Accept zero-delta moves to explore
                colors[i], colors[j] = colors[j], colors[i]
            else:
                if T > 0.01:
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        colors[i], colors[j] = colors[j], colors[i]
                        current_cost += delta
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_colors = colors.copy()
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        return best_colors, best_cost
    
    # Global best
    global_best_colors = None
    global_best_cost = float('inf')
    
    # Multiple restarts
    num_restarts = 12
    time_per_restart = 130.0 / num_restarts  # leave 10s for final polish
    
    initial_strategies = [
        lambda: np.zeros(N, dtype=np.int8),
        lambda: np.ones(N, dtype=np.int8),
        lambda: np.array([i % 2 for i in range(N)], dtype=np.int8),
        lambda: np.array([(i // 4) % 2 for i in range(N)], dtype=np.int8),
        lambda: np.array([(i % 4) >= 2 for i in range(N)], dtype=np.int8),
        lambda: np.array([(i // 8) % 2 for i in range(N)], dtype=np.int8),
        lambda: np.array([(i * 7) % 5 < 2 for i in range(N)], dtype=np.int8),
        lambda: np.array([(i * 13) % 7 < 3 for i in range(N)], dtype=np.int8),
        lambda: np.random.randint(0, 2, N).astype(np.int8),
        lambda: np.random.randint(0, 2, N).astype(np.int8),
        lambda: np.array([(i % 3 == 0) for i in range(N)], dtype=np.int8),
        lambda: np.array([(i % 6 < 3) for i in range(N)], dtype=np.int8),
    ]
    
    for restart in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        if restart < len(initial_strategies):
            initial = initial_strategies[restart]()
        elif global_best_colors is not None:
            # Perturb best
            initial = global_best_colors.copy()
            for _ in range(N // 4):
                idx = random.randint(0, N - 1)
                initial[idx] = 1 - initial[idx]
        else:
            initial = np.random.randint(0, 2, N).astype(np.int8)
        
        # Vary SA parameters
        T_start = 1.5 + 0.5 * (restart % 3)
        cooling = 0.9998 - 0.00002 * (restart % 5)
        min_T = 0.01
        
        sa_colors, sa_cost = simulated_annealing_with_swaps(
            initial, T_start, cooling, min_T, time_per_restart * 0.7
        )
        
        # Pair-swap polish
        polished_colors, polished_cost = pair_swap_polish(
            sa_colors, sa_cost, max_sweeps=30
        )
        
        # Also try single-flip polish on the polished result
        # (single flips can still help after pair swaps)
        for _ in range(3):
            improved = False
            for i in range(N):
                if time.time() > deadline - 5:
                    break
                # Compute delta for single flip at i
                c_i = polished_colors[i]
                d_flip = 0
                for ap_idx in aps_at[i]:
                    ap = ap_array[ap_idx]
                    others = [ap[k] for k in range(4) if ap[k] != i]
                    o_colors = polished_colors[others]
                    if len(set(o_colors.tolist())) == 1:
                        if c_i == o_colors[0]:
                            d_flip -= 1
                        else:
                            d_flip += 1
                if d_flip < 0:
                    polished_colors[i] = 1 - polished_colors[i]
                    polished_cost += d_flip
                    improved = True
            if not improved:
                break
        
        if polished_cost < global_best_cost:
            global_best_cost = polished_cost
            global_best_colors = polished_colors.copy()
    
    # Final polish
    if global_best_colors is not None:
        final_colors, final_cost = pair_swap_polish(
            global_best_colors, global_best_cost, max_sweeps=20
        )
        if final_cost < global_best_cost:
            global_best_cost = final_cost
            global_best_colors = final_colors
    
    # Convert to list of ints
    if global_best_colors is not None:
        return [int(x) for x in global_best_colors]
    else:
        return [0] * N
