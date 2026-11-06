import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 130  # Safety margin before 150s kill
    
    # Precompute all 4-term APs
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            p0 = a - 1
            p1 = a + d - 1
            p2 = a + 2 * d - 1
            p3 = a + 3 * d - 1
            aps.append((p0, p1, p2, p3))
    
    num_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)
    
    # Build position -> AP index mapping (as lists for fast iteration)
    pos_aps = [[] for _ in range(n)]
    for idx in range(num_aps):
        p0, p1, p2, p3 = aps[idx]
        pos_aps[p0].append((idx, 0))
        pos_aps[p1].append((idx, 1))
        pos_aps[p2].append((idx, 2))
        pos_aps[p3].append((idx, 3))
    
    # Convert to numpy for vectorized operations
    # For each position, store the AP indices and offsets
    pos_ap_indices = []
    pos_ap_offsets = []
    for p in range(n):
        if pos_aps[p]:
            indices = [x[0] for x in pos_aps[p]]
            offsets = [x[1] for x in pos_aps[p]]
            pos_ap_indices.append(np.array(indices, dtype=np.int32))
            pos_ap_offsets.append(np.array(offsets, dtype=np.int32))
        else:
            pos_ap_indices.append(np.array([], dtype=np.int32))
            pos_ap_offsets.append(np.array([], dtype=np.int32))
    
    def count_mono(color):
        c = np.array(color, dtype=np.int8)
        c0 = c[aps_arr[:, 0]]
        c1 = c[aps_arr[:, 1]]
        c2 = c[aps_arr[:, 2]]
        c3 = c[aps_arr[:, 3]]
        return int(np.sum((c0 == c1) & (c1 == c2) & (c2 == c3)))
    
    def init_mono_state(color):
        c = np.array(color, dtype=np.int8)
        c0 = c[aps_arr[:, 0]]
        c1 = c[aps_arr[:, 1]]
        c2 = c[aps_arr[:, 2]]
        c3 = c[aps_arr[:, 3]]
        mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        count = int(np.sum(mono))
        return mono, count
    
    def greedy_descent(color, mono, count, max_iter=100):
        """Fast greedy descent using vectorized delta computation."""
        c = np.array(color, dtype=np.int8)
        
        for _ in range(max_iter):
            if time.time() > deadline - 3:
                break
            
            best_delta = 0
            best_pos = -1
            
            for p in range(n):
                if len(pos_ap_indices[p]) == 0:
                    continue
                
                ap_idx = pos_ap_indices[p]
                offset = pos_ap_offsets[p]
                new_val = 1 - c[p]
                
                # Get the other 3 positions for each AP
                # For AP at index i, the positions are aps_arr[i]
                other_pos = aps_arr[ap_idx].copy()
                # Replace the position p with -1 (we'll handle it)
                # Actually, we need to check if the other 3 are all new_val
                
                # Get colors of all 4 positions for these APs
                all_colors = c[aps_arr[ap_idx]]  # shape (num_aps_for_p, 4)
                
                # The flipped position will have new_val
                # Check: are the other 3 all equal to new_val?
                # For each AP, the offset tells us which position is p
                # After flip, position at offset has new_val
                # We need: all other 3 positions == new_val
                
                # Create a copy where the flipped position is set to new_val
                new_colors = all_colors.copy()
                new_colors[np.arange(len(ap_idx)), offset] = new_val
                
                is_mono = (new_colors[:, 0] == new_colors[:, 1]) & \
                          (new_colors[:, 1] == new_colors[:, 2]) & \
                          (new_colors[:, 2] == new_colors[:, 3])
                
                was_mono = mono[ap_idx]
                
                # Delta: +1 for each AP that becomes mono, -1 for each that stops being mono
                becomes_mono = is_mono & ~was_mono
                stops_mono = ~is_mono & was_mono
                
                delta = int(np.sum(becomes_mono)) - int(np.sum(stops_mono))
                
                if delta < best_delta:
                    best_delta = delta
                    best_pos = p
            
            if best_pos != -1:
                # Apply the flip
                p = best_pos
                new_val = 1 - c[p]
                ap_idx = pos_ap_indices[p]
                offset = pos_ap_offsets[p]
                
                all_colors = c[aps_arr[ap_idx]]
                new_colors = all_colors.copy()
                new_colors[np.arange(len(ap_idx)), offset] = new_val
                
                is_mono = (new_colors[:, 0] == new_colors[:, 1]) & \
                          (new_colors[:, 1] == new_colors[:, 2]) & \
                          (new_colors[:, 2] == new_colors[:, 3])
                
                mono[ap_idx] = is_mono
                c[p] = new_val
                count += best_delta
            else:
                break
        
        return color, mono, count
    
    def sa_search(color, mono, count, T0=2.0, cooling=0.995, max_iter=5000):
        """Simulated annealing with vectorized delta computation."""
        c = np.array(color, dtype=np.int8)
        T = T0
        min_T = 1e-4
        
        for _ in range(max_iter):
            if time.time() > deadline - 3:
                break
            if T < min_T:
                break
            
            p = random.randint(0, n - 1)
            
            if len(pos_ap_indices[p]) == 0:
                continue
            
            new_val = 1 - c[p]
            ap_idx = pos_ap_indices[p]
            offset = pos_ap_offsets[p]
            
            all_colors = c[aps_arr[ap_idx]]
            new_colors = all_colors.copy()
            new_colors[np.arange(len(ap_idx)), offset] = new_val
            
            is_mono = (new_colors[:, 0] == new_colors[:, 1]) & \
                      (new_colors[:, 1] == new_colors[:, 2]) & \
                      (new_colors[:, 2] == new_colors[:, 3])
            
            was_mono = mono[ap_idx]
            
            becomes_mono = is_mono & ~was_mono
            stops_mono = ~is_mono & was_mono
            
            delta = int(np.sum(becomes_mono)) - int(np.sum(stops_mono))
            
            # Accept if improving or with probability exp(-delta/T)
            if delta < 0 or random.random() < np.exp(-delta / T):
                mono[ap_idx] = is_mono
                c[p] = new_val
                count += delta
            
            T *= cooling
        
        return color, mono, count
    
    def tabu_search(color, mono, count, tabu_len=20, max_iter=2000):
        """Tabu search to escape local optima."""
        c = np.array(color, dtype=np.int8)
        tabu = [0] * n  # tabu[p] = iteration when p becomes untabu
        cur_iter = 0
        best_count = count
        best_color = color[:]
        
        for _ in range(max_iter):
            if time.time() > deadline - 3:
                break
            
            cur_iter += 1
            best_delta = float('inf')
            best_pos = -1
            best_is_tabu = False
            
            for p in range(n):
                if len(pos_ap_indices[p]) == 0:
                    continue
                
                new_val = 1 - c[p]
                ap_idx = pos_ap_indices[p]
                offset = pos_ap_offsets[p]
                
                all_colors = c[aps_arr[ap_idx]]
                new_colors = all_colors.copy()
                new_colors[np.arange(len(ap_idx)), offset] = new_val
                
                is_mono = (new_colors[:, 0] == new_colors[:, 1]) & \
                          (new_colors[:, 1] == new_colors[:, 2]) & \
                          (new_colors[:, 2] == new_colors[:, 3])
                
                was_mono = mono[ap_idx]
                
                becomes_mono = is_mono & ~was_mono
                stops_mono = ~is_mono & was_mono
                
                delta = int(np.sum(becomes_mono)) - int(np.sum(stops_mono))
                
                is_tabu = tabu[p] > cur_iter
                
                # Aspiration: accept tabu move if it improves best
                if is_tabu and delta < best_count:
                    is_tabu = False
                
                if not is_tabu and delta < best_delta:
                    best_delta = delta
                    best_pos = p
                    best_is_tabu = False
            
            if best_pos != -1:
                p = best_pos
                new_val = 1 - c[p]
                ap_idx = pos_ap_indices[p]
                offset = pos_ap_offsets[p]
                
                all_colors = c[aps_arr[ap_idx]]
                new_colors = all_colors.copy()
                new_colors[np.arange(len(ap_idx)), offset] = new_val
                
                is_mono = (new_colors[:, 0] == new_colors[:, 1]) & \
                          (new_colors[:, 1] == new_colors[:, 2]) & \
                          (new_colors[:, 2] == new_colors[:, 3])
                
                mono[ap_idx] = is_mono
                c[p] = new_val
                count += best_delta
                tabu[p] = cur_iter + tabu_len
                
                if count < best_count:
                    best_count = count
                    best_color = color[:]
            else:
                break
        
        if count < best_count:
            best_color = color[:]
        
        return best_color, mono, best_count
    
    # Initializations
    def thue_morse(n):
        return [bin(i).count('1') % 2 for i in range(n)]
    
    def random_init(n):
        return [random.randint(0, 1) for _ in range(n)]
    
    def striped(n):
        # 010101...
        return [i % 2 for i in range(n)]
    
    def block_pattern(n, block_size):
        # 000...0111...1000...0111...1
        return [(i // block_size) % 2 for i in range(n)]
    
    def sine_pattern(n):
        # Based on sin
        return [1 if np.sin(i * 0.1) > 0 else 0 for i in range(n)]
    
    def fourier_pattern(n):
        # Sum of a few sinusoids
        pattern = []
        for i in range(n):
            val = np.sin(i * 0.1) + 0.5 * np.sin(i * 0.3) + 0.3 * np.sin(i * 0.7)
            pattern.append(1 if val > 0 else 0)
        return pattern
    
    random.seed(12345)
    
    best_color = None
    best_score = float('inf')
    
    initializers = [
        thue_morse,
        lambda: [1 - x for x in thue_morse(n)],
        random_init,
        striped,
        lambda: block_pattern(n, 2),
        lambda: block_pattern(n, 4),
        lambda: block_pattern(n, 8),
        lambda: block_pattern(n, 16),
        sine_pattern,
        fourier_pattern,
    ]
    
    # Phase 1: Try each initializer with greedy + SA
    for init_func in initializers:
        if time.time() > deadline - 15:
            break
        
        color = init_func(n)
        mono, count = init_mono_state(color)
        
        # Greedy descent
        color, mono, count = greedy_descent(color, mono, count, max_iter=200)
        
        # SA
        color, mono, count = sa_search(color, mono, count, T0=1.5, cooling=0.997, max_iter=3000)
        
        if count < best_score:
            best_score = count
            best_color = color[:]
    
    # Phase 2: ILS from best solution
    base_color = best_color[:]
    base_mono, base_count = init_mono_state(base_color)
    
    for restart in range(50):
        if time.time() > deadline - 5:
            break
        
        # Perturb
        perturbed = base_color[:]
        num_flips = random.randint(10, 30)
        for _ in range(num_flips):
            p = random.randint(0, n - 1)
            perturbed[p] = 1 - perturbed[p]
        
        p_mono, p_count = init_mono_state(perturbed)
        
        # Greedy + SA
        p_color, p_mono, p_count = greedy_descent(perturbed, p_mono, p_count, max_iter=150)
        p_color, p_mono, p_count = sa_search(p_color, p_mono, p_count, T0=1.0, cooling=0.996, max_iter=2000)
        
        # Tabu from best
        t_color, t_mono, t_count = tabu_search(p_color, p_mono, p_count, tabu_len=15, max_iter=500)
        
        if t_count < best_score:
            best_score = t_count
            best_color = t_color[:]
            base_color = t_color[:]
            base_mono, base_count = init_mono_state(base_color)
        elif p_count < best_score:
            best_score = p_count
            best_color = p_color[:]
            base_color = p_color[:]
            base_mono, base_count = init_mono_state(base_color)
    
    # Phase 3: Final polish with more SA iterations
    if time.time() < deadline - 10:
        final_mono, final_count = init_mono_state(best_color)
        final_color, final_mono, final_count = sa_search(best_color, final_mono, final_count, T0=0.5, cooling=0.998, max_iter=5000)
        if final_count < best_score:
            best_score = final_count
            best_color = final_color[:]
    
    return best_color

