import time
import random
import numpy as np

def construct():
    n = 256
    deadline = time.time() + 110
    
    # Precompute all 4-APs: (a, a+d, a+2d, a+3d) with 0-indexed
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(0, n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)
    
    # For each position, list of (ap_index, position_in_ap)
    pos_aps = [[] for _ in range(n)]
    for idx in range(num_aps):
        for j in range(4):
            pos = aps[idx][j]
            pos_aps[pos].append((idx, j))
    
    # Precompute for fast vectorized delta computation
    # For each position, store the AP indices and which slot (0-3) it occupies
    pos_ap_indices = []
    pos_ap_slots = []
    for pos in range(n):
        idxs = [ap_idx for ap_idx, _ in pos_aps[pos]]
        slots = [slot for _, slot in pos_aps[pos]]
        pos_ap_indices.append(np.array(idxs, dtype=np.int32))
        pos_ap_slots.append(np.array(slots, dtype=np.int32))
    
    def count_mono_np(color_arr):
        """Count monochromatic 4-APs using numpy."""
        c = color_arr[aps_arr]
        mono = (c[:, 0] == c[:, 1]) & (c[:, 1] == c[:, 2]) & (c[:, 2] == c[:, 3])
        return int(mono.sum())
    
    def fast_delta_flip(color_arr, pos):
        """
        Vectorized delta computation for flipping position pos.
        Returns the change in mono count.
        """
        ap_idxs = pos_ap_indices[pos]
        slots = pos_ap_slots[pos]
        if len(ap_idxs) == 0:
            return 0
        
        # Get all APs containing this position
        sub_aps = aps_arr[ap_idxs]  # shape (k, 4)
        sub_slots = slots  # shape (k,)
        
        # Current colors of these APs
        cur_colors = color_arr[sub_aps]  # shape (k, 4)
        
        # Old color of this position
        old_c = color_arr[pos]
        new_c = 1 - old_c
        
        # For each AP, check if the other 3 positions are all equal to old_c (mono before)
        # or all equal to new_c (mono after)
        
        # Create mask for the position we're flipping
        # For each AP row, we need to check the 3 other positions
        
        # Initialize: was_mono_before = False, will_mono_after = False
        was_mono = np.zeros(len(ap_idxs), dtype=bool)
        will_mono = np.zeros(len(ap_idxs), dtype=bool)
        
        for slot in range(4):
            mask = (sub_slots == slot)
            if not np.any(mask):
                continue
            # For APs where this position is at slot
            rows = np.where(mask)[0]
            # Get the other 3 positions
            other_slots = [s for s in range(4) if s != slot]
            other_colors = cur_colors[rows][:, other_slots]  # shape (num_rows, 3)
            
            # Check if all 3 others equal old_c
            all_old = np.all(other_colors == old_c, axis=1)
            was_mono[rows] = all_old
            
            # Check if all 3 others equal new_c
            all_new = np.all(other_colors == new_c, axis=1)
            will_mono[rows] = all_new
        
        delta = int(np.sum(will_mono) - np.sum(was_mono))
        return delta
    
    def local_search(color, max_iters=500):
        """Greedy local search with fast delta computation."""
        color_arr = np.array(color, dtype=np.int8)
        cur_count = count_mono_np(color_arr)
        improved = True
        iters = 0
        while improved and iters < max_iters and time.time() < deadline:
            improved = False
            iters += 1
            best_delta = 0
            best_pos = -1
            
            for pos in range(n):
                d = fast_delta_flip(color_arr, pos)
                if d < best_delta:
                    best_delta = d
                    best_pos = pos
            
            if best_pos >= 0:
                color_arr[best_pos] = 1 - color_arr[best_pos]
                cur_count += best_delta
                improved = True
        
        return color_arr.tolist(), cur_count
    
    def simulated_annealing(color, max_iters=5000, t0=3.0, t1=0.05):
        """Simulated annealing with fast delta computation."""
        color_arr = np.array(color, dtype=np.int8)
        cur_count = count_mono_np(color_arr)
        best_color = color_arr.copy()
        best_count = cur_count
        t = t0
        cooling = (t1 / t0) ** (1.0 / max_iters)
        
        for it in range(max_iters):
            if time.time() > deadline:
                break
            pos = random.randint(0, n - 1)
            d = fast_delta_flip(color_arr, pos)
            new_count = cur_count + d
            if new_count < cur_count or (d > 0 and random.random() < np.exp(-d / max(t, 1e-10))):
                color_arr[pos] = 1 - color_arr[pos]
                cur_count = new_count
                if cur_count < best_count:
                    best_count = cur_count
                    best_color = color_arr.copy()
            t *= cooling
        
        return best_color.tolist(), best_count
    
    def thue_morse(n):
        return [bin(i).count('1') % 2 for i in range(n)]
    
    def random_color(n):
        return [random.randint(0, 1) for _ in range(n)]
    
    # Multi-restart local search with diverse initializations
    best_color = None
    best_score = float('inf')
    
    # Generate diverse starting points
    initial_colors = []
    
    # Thue-Morse and complement
    tm = thue_morse(n)
    initial_colors.append(tm)
    initial_colors.append([1 - x for x in tm])
    
    # Structured patterns
    for block in [2, 3, 4, 5, 7, 8, 11, 13]:
        initial_colors.append([(i // block) % 2 for i in range(n)])
    
    # Random initializations
    for _ in range(8):
        initial_colors.append(random_color(n))
    
    # Evaluate all and keep track of best
    for init in initial_colors:
        if time.time() > deadline - 5:
            break
        polished, sc = local_search(init, max_iters=300)
        if sc < best_score:
            best_score = sc
            best_color = polished[:]
    
    # If we have time, do more restarts from perturbed best
    num_extra_restarts = 0
    while time.time() < deadline - 10:
        num_extra_restarts += 1
        # Perturb best solution
        perturbed = best_color[:]
        num_flips = random.randint(3, 8)
        flip_positions = random.sample(range(n), num_flips)
        for pos in flip_positions:
            perturbed[pos] = 1 - perturbed[pos]
        
        polished, sc = local_search(perturbed, max_iters=200)
        if sc < best_score:
            best_score = sc
            best_color = polished[:]
    
    # Final SA polish on best
    if time.time() < deadline - 5:
        sa_color, sa_score = simulated_annealing(best_color, max_iters=3000)
        if sa_score < best_score:
            best_color = sa_color
            best_score = sa_score
    
    # Final local search
    if time.time() < deadline - 2:
        polished, sc = local_search(best_color, max_iters=300)
        if sc < best_score:
            best_color = polished
            best_score = sc
    
    return best_color
