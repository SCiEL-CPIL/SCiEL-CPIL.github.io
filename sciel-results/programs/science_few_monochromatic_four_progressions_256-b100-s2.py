"""Program using multi-start simulated annealing with diverse structured initializations."""
import numpy as np
import random
import time

def count_mono_4ap(colors, n=256):
    """Count monochromatic 4-term arithmetic progressions in colors[0..n-1]."""
    count = 0
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            c0 = colors[a]
            if c0 == colors[a + d] == colors[a + 2 * d] == colors[a + 3 * d]:
                count += 1
    return count

def construct():
    n = 256
    
    # Precompute APs and position-to-AP mapping for fast delta calculation
    ap_list = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    pos_to_aps = [[] for _ in range(n)]
    for idx, (a, b, c, d_pos) in enumerate(ap_list):
        pos_to_aps[a].append(idx)
        pos_to_aps[b].append(idx)
        pos_to_aps[c].append(idx)
        pos_to_aps[d_pos].append(idx)

    def make_thue_morse():
        """Thue-Morse sequence: t(n) = popcount(n) % 2."""
        colors = []
        for i in range(n):
            colors.append(bin(i).count('1') % 2)
        return colors

    def make_block_coloring(block_size):
        colors = []
        for i in range(n):
            colors.append((i // block_size) % 2)
        return colors
    
    def make_mod_coloring(mod):
        colors = []
        for i in range(n):
            colors.append((i % mod) % 2)
        return colors

    def make_xor_pattern(x):
        """Color based on XOR of block index and sub-block index."""
        colors = []
        for i in range(n):
            block = i // x
            sub = (i % x) // (x // 2)
            colors.append(block % 2 ^ sub)
        return colors

    # Generate diverse structured candidates
    candidates = []
    
    # Thue-Morse and inverted
    candidates.append(make_thue_morse())
    tm = make_thue_morse()
    candidates.append([1 - c for c in tm])
    
    # Block sizes
    for bs in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 32, 64]:
        candidates.append(make_block_coloring(bs))
        
    # Mod-based
    for mod in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 32]:
        candidates.append(make_mod_coloring(mod))
        
    # XOR patterns
    for x in [4, 8, 16, 32, 64]:
        candidates.append(make_xor_pattern(x))
        
    # Random initializations (fewer, since structured ones are better)
    for _ in range(15):
        candidates.append([random.randint(0, 1) for _ in range(n)])
        
    # Find best initial coloring
    best_colors_init = None
    best_count_init = float('inf')
    for cand in candidates:
        c = count_mono_4ap(cand, n)
        if c < best_count_init:
            best_count_init = c
            best_colors_init = cand[:]
            
    # Multi-Start Simulated Annealing
    start_time = time.time()
    time_limit = 115  # seconds
    
    global_best_colors = best_colors_init[:]
    global_best_count = best_count_init
    
    num_starts = 12
    time_per_start = (time_limit - 5) / num_starts
    
    for start_idx in range(num_starts):
        if time.time() - start_time > time_limit - 2:
            break
            
        # Initialize current state
        if start_idx == 0:
            colors = global_best_colors[:]
        else:
            # Perturb the global best by flipping a few random bits
            colors = global_best_colors[:]
            num_flips = random.randint(3, 15)
            for _ in range(num_flips):
                pos = random.randint(0, n - 1)
                colors[pos] = 1 - colors[pos]
        
        current_count = count_mono_4ap(colors, n)
        
        # SA Parameters
        T = 5.0
        T_min = 0.001
        max_iters = 150000
        cooling = (T_min / T) ** (1.0 / max_iters)
        
        for i in range(max_iters):
            if time.time() - start_time > time_limit - 1:
                break
                
            pos = random.randint(0, n - 1)
            old_val = colors[pos]
            new_val = 1 - old_val
            
            delta = 0
            for ap_idx in pos_to_aps[pos]:
                a, b, c, d_pos = ap_list[ap_idx]
                
                # Count matches of old_val among the other 3 positions
                match_old = 0
                if a == pos:
                    if colors[b] == old_val: match_old += 1
                    if colors[c] == old_val: match_old += 1
                    if colors[d_pos] == old_val: match_old += 1
                elif b == pos:
                    if colors[a] == old_val: match_old += 1
                    if colors[c] == old_val: match_old += 1
                    if colors[d_pos] == old_val: match_old += 1
                elif c == pos:
                    if colors[a] == old_val: match_old += 1
                    if colors[b] == old_val: match_old += 1
                    if colors[d_pos] == old_val: match_old += 1
                else:  # d_pos == pos
                    if colors[a] == old_val: match_old += 1
                    if colors[b] == old_val: match_old += 1
                    if colors[c] == old_val: match_old += 1
                
                # mono_before if match_old == 3
                # mono_after if match_old == 0
                if match_old == 3:
                    delta -= 1
                elif match_old == 0:
                    delta += 1
            
            if delta <= 0:
                colors[pos] = new_val
                current_count += delta
                if current_count < global_best_count:
                    global_best_count = current_count
                    global_best_colors = colors[:]
            else:
                if random.random() < np.exp(-delta / T):
                    colors[pos] = new_val
                    current_count += delta
            
            T *= cooling
            if T < T_min:
                T = T_min

    return global_best_colors
