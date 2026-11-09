import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    n = 256

    # Precompute all 4-term APs: (i0, i1, i2, i3) 0-indexed
    # a, a+d, a+2d, a+3d with 1<=a, d>=1, a+3d<=256
    aps = []
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps.append((i0, i1, i2, i3))
    
    aps_arr = np.array(aps, dtype=np.int32)
    M = len(aps_arr)
    
    # Build per-position list of AP indices
    # pos_aps[i] = list of AP indices where position i appears
    pos_aps = [[] for _ in range(n)]
    for idx, (i0, i1, i2, i3) in enumerate(aps):
        pos_aps[i0].append(idx)
        pos_aps[i1].append(idx)
        pos_aps[i2].append(idx)
        pos_aps[i3].append(idx)
    
    # Convert to numpy arrays for speed
    pos_aps_arr = [np.array(p, dtype=np.int32) for p in pos_aps]
    
    # Precompute "others" for each position to speed up flip_delta
    # pos_others[p] = array of shape (K, 3) containing the other 3 positions for each AP at p
    pos_others = [None] * n
    for p in range(n):
        aps_at_p = pos_aps_arr[p]
        if len(aps_at_p) == 0:
            pos_others[p] = np.empty((0, 3), dtype=np.int32)
            continue
        all_indices = aps_arr[aps_at_p]  # shape (K, 4)
        not_p_mask = (all_indices != p)
        rows, cols = np.where(not_p_mask)
        others_vals = all_indices[rows, cols]
        pos_others[p] = others_vals.reshape(-1, 3)
    
    # Count monochromatic APs for a coloring (0/1 array)
    def count_mono(color):
        # For each AP, check if all 4 positions have same color
        # color[aps_arr[k]] gives colors at the 4 positions
        cols = color[aps_arr]  # shape (M, 4)
        # Monochromatic if all 4 are equal: min==max
        mono = (cols.min(axis=1) == cols.max(axis=1))
        return int(mono.sum())
    
    # Compute delta in mono count if we flip position p
    # Vectorized version using precomputed pos_others
    def flip_delta(color, p):
        """Return change in mono count if we flip color[p]."""
        others = pos_others[p]
        if len(others) == 0:
            return 0
        
        # Get colors at the other 3 positions for all APs containing p
        # others has shape (K, 3), so color[others] gives shape (K, 3)
        other_colors = color[others]  # shape (K, 3)
        
        # Current color at p
        cp = color[p]
        new_cp = 1 - cp
        
        # Current state: AP is mono if all 3 others equal cp
        # other_colors has shape (K, 3), check if all equal cp
        all_same_now = np.all(other_colors == cp, axis=1)  # shape (K,)
        
        # New state: AP is mono after flip if all 3 others equal new_cp
        all_same_new = np.all(other_colors == new_cp, axis=1)  # shape (K,)
        
        # Count transitions
        # +1 if not mono now but mono after (all_same_now=False, all_same_new=True)
        # -1 if mono now but not mono after (all_same_now=True, all_same_new=False)
        delta = np.sum((~all_same_now) & all_same_new) - np.sum(all_same_now & (~all_same_new))
        return int(delta)
    
    # Thue-Morse initial coloring
    def thue_morse(n):
        color = np.zeros(n, dtype=np.int8)
        for i in range(n):
            # Count bits in i
            color[i] = bin(i).count('1') % 2
        return color
    
    # Generate several diverse starting points
    def random_coloring():
        return np.random.randint(0, 2, n).astype(np.int8)
    
    def block_coloring(block_size):
        color = np.zeros(n, dtype=np.int8)
        for i in range(n):
            color[i] = (i // block_size) % 2
        return color
    
    # Local search: hill climbing with best-improvement (more thorough)
    def local_search(color, max_iter=5000):
        color = color.copy()
        current_score = count_mono(color)
        improved = True
        while improved and max_iter > 0:
            improved = False
            max_iter -= 1
            # Try flipping each position, find best improvement
            best_delta = 0
            best_pos = -1
            for p in range(n):
                d = flip_delta(color, p)
                if d < best_delta:
                    best_delta = d
                    best_pos = p
            if best_pos >= 0:
                color[best_pos] = 1 - color[best_pos]
                current_score += best_delta
                improved = True
            if not improved:
                break
        return color, current_score

    # Simulated Annealing to escape local minima
    def simulated_annealing(color, max_iter=500, start_temp=2.0, end_temp=0.1):
        color = color.copy()
        current_score = count_mono(color)
        best_score = current_score
        best_color = color.copy()
        
        for i in range(max_iter):
            if time.time() - start_time > deadline - 5:
                break
            
            # Linear cooling or geometric? Geometric is standard.
            # T = T_start * (T_end/T_start)^(i/max_iter)
            ratio = i / max_iter
            temp = start_temp * (end_temp / start_temp) ** ratio
            
            p = np.random.randint(0, n)
            delta = flip_delta(color, p)
            
            # Accept if improvement or with probability exp(-delta/T)
            if delta < 0 or np.random.random() < np.exp(-delta / temp):
                color[p] = 1 - color[p]
                current_score += delta
                
                if current_score < best_score:
                    best_score = current_score
                    best_color = color.copy()
                    
        return best_color, best_score

    # Iterated local search with SA and HC
    def iterated_local_search(color, max_iterations=200):
        color = color.copy()
        best_color = color.copy()
        best_score = count_mono(color)
        
        for it in range(max_iterations):
            # Check time
            if time.time() - start_time > deadline - 5:
                break
            
            # Phase 1: Simulated Annealing to explore
            color, score = simulated_annealing(color, max_iter=200)
            
            # Phase 2: Hill Climbing to exploit
            color, score = local_search(color, max_iter=1000)
            
            if score < best_score:
                best_score = score
                best_color = color.copy()
            
            # Small perturbation to ensure diversity
            num_flips = np.random.randint(1, 4)
            flip_positions = np.random.choice(n, size=min(num_flips, n), replace=False)
            color[flip_positions] = 1 - color[flip_positions]
            
        return best_color, best_score
    
    # Generate diverse starting points
    startings = [thue_morse(n)]
    for bs in [1, 2, 3, 4, 5, 6, 7, 8, 16, 32]:
        startings.append(block_coloring(bs))
    for _ in range(8):
        startings.append(random_coloring())
    
    # Also try Thue-Morse with random perturbations
    for _ in range(5):
        tm = thue_morse(n).copy()
        flip_pos = np.random.choice(n, size=10, replace=False)
        tm[flip_pos] = 1 - tm[flip_pos]
        startings.append(tm)
    
    best_overall_color = None
    best_overall_score = float('inf')
    
    for start_color in startings:
        if time.time() - start_time > deadline - 10:
            break
        
        # Run iterated local search
        color, score = iterated_local_search(start_color, max_iterations=50)
        
        if score < best_overall_score:
            best_overall_score = score
            best_overall_color = color.copy()
    
    # If we have time, do more iterations on the best
    if time.time() - start_time < deadline - 20 and best_overall_color is not None:
        color, score = iterated_local_search(best_overall_color, max_iterations=200)
        if score < best_overall_score:
            best_overall_score = score
            best_overall_color = color.copy()
    
    if best_overall_color is None:
        best_overall_color = thue_morse(n)
    
    return [int(c) for c in best_overall_color]
