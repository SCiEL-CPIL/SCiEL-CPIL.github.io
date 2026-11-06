"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    n = 256
    
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) with 0-indexed
    ap_list = []
    for d in range(1, (n + 2) // 3 + 1):
        for i in range(n - 3 * d):
            ap_list.append((i, i + d, i + 2 * d, i + 3 * d))
    
    num_aps = len(ap_list)
    
    # AP position arrays for fast vectorized computation
    ap_a = np.array([x[0] for x in ap_list], dtype=np.int32)
    ap_b = np.array([x[1] for x in ap_list], dtype=np.int32)
    ap_c = np.array([x[2] for x in ap_list], dtype=np.int32)
    ap_d = np.array([x[3] for x in ap_list], dtype=np.int32)
    
    # For each index, precompute the AP indices it participates in
    index_to_aps = [[] for _ in range(n)]
    for ap_idx in range(num_aps):
        index_to_aps[ap_a[ap_idx]].append(ap_idx)
        index_to_aps[ap_b[ap_idx]].append(ap_idx)
        index_to_aps[ap_c[ap_idx]].append(ap_idx)
        index_to_aps[ap_d[ap_idx]].append(ap_idx)
    
    # Convert to numpy arrays for speed
    index_to_aps_arr = [np.array(lst, dtype=np.int32) for lst in index_to_aps]
    
    def count_mono(coloring):
        """Count monochromatic 4-term APs."""
        mono = (coloring[ap_a] == coloring[ap_b]) & \
               (coloring[ap_b] == coloring[ap_c]) & \
               (coloring[ap_c] == coloring[ap_d])
        return int(np.sum(mono))
    
    def flip_delta_fast(coloring, idx):
        """Compute delta for flipping position idx using vectorized ops."""
        aps = index_to_aps_arr[idx]
        if len(aps) == 0:
            return 0
        a = ap_a[aps]
        b = ap_b[aps]
        c = ap_c[aps]
        d = ap_d[aps]
        ca = coloring[a]
        cb = coloring[b]
        cc = coloring[c]
        cd = coloring[d]
        # Current mono status
        was_mono = (ca == cb) & (cb == cc) & (cc == cd)
        # After flip: replace color at idx with 1-coloring[idx]
        new_col = 1 - coloring[idx]
        na = np.where(a == idx, new_col, ca)
        nb = np.where(b == idx, new_col, cb)
        nc = np.where(c == idx, new_col, cc)
        nd = np.where(d == idx, new_col, cd)
        is_mono_now = (na == nb) & (nb == nc) & (nc == nd)
        # Delta: -1 for each mono->non-mono, +1 for each non-mono->mono
        delta = int(np.sum(is_mono_now & ~was_mono)) - int(np.sum(~is_mono_now & was_mono))
        return delta
    
    def flip_delta_two(coloring, idx1, idx2):
        """Compute delta for flipping two positions simultaneously."""
        # Union of APs involving either idx1 or idx2
        set1 = set(index_to_aps[idx1])
        set2 = set(index_to_aps[idx2])
        union = list(set1 | set2)
        if not union:
            return 0
        aps = np.array(union, dtype=np.int32)
        a = ap_a[aps]
        b = ap_b[aps]
        c = ap_c[aps]
        d = ap_d[aps]
        ca = coloring[a]
        cb = coloring[b]
        cc = coloring[c]
        cd = coloring[d]
        was_mono = (ca == cb) & (cb == cc) & (cc == cd)
        nc1 = 1 - coloring[idx1]
        nc2 = 1 - coloring[idx2]
        # Apply both flips
        na = np.where(a == idx1, nc1, ca)
        na = np.where(a == idx2, nc2, na)
        nb = np.where(b == idx1, nc1, cb)
        nb = np.where(b == idx2, nc2, nb)
        nc = np.where(c == idx1, nc1, cc)
        nc = np.where(c == idx2, nc2, nc)
        nd = np.where(d == idx1, nc1, cd)
        nd = np.where(d == idx2, nc2, nd)
        is_mono_now = (na == nb) & (nb == nc) & (nc == nd)
        delta = int(np.sum(is_mono_now & ~was_mono)) - int(np.sum(~is_mono_now & was_mono))
        return delta
    
    # Generate initial colorings
    def thue_morse():
        colors = np.zeros(n, dtype=np.int8)
        for i in range(n):
            colors[i] = bin(i).count('1') % 2
        return colors
    
    def block_pattern(block_size):
        colors = np.zeros(n, dtype=np.int8)
        for i in range(n):
            colors[i] = (i // block_size) % 2
        return colors
    
    def mod_pattern(m):
        colors = np.zeros(n, dtype=np.int8)
        for i in range(n):
            colors[i] = (i % m) % 2
        return colors
    
    # Multi-phase search
    best_coloring = None
    best_count = float('inf')
    
    # Phase 1: Try several structured initializations
    initializations = [
        thue_morse(),
        block_pattern(2),
        block_pattern(3),
        block_pattern(4),
        block_pattern(5),
        block_pattern(6),
        mod_pattern(3),
        mod_pattern(4),
        mod_pattern(5),
        mod_pattern(7),
        np.random.randint(0, 2, n).astype(np.int8),
        np.random.randint(0, 2, n).astype(np.int8),
    ]
    
    for init in initializations:
        cnt = count_mono(init)
        if cnt < best_count:
            best_count = cnt
            best_coloring = init.copy()
    
    # Phase 2: Simulated Annealing from best initialization
    # Use faster delta; more iterations possible
    coloring = best_coloring.copy()
    current_count = best_count
    T = 1.0
    cooling_rate = 0.9998
    min_T = 0.005
    
    # Pre-generate random indices in batches for speed
    batch_size = 1000
    while time.time() < deadline - 35:
        if T < min_T:
            T = min_T
            break
        
        # Batch of random indices
        batch = np.random.randint(0, n, batch_size)
        for idx in batch:
            if time.time() > deadline - 35:
                break
            d = flip_delta_fast(coloring, idx)
            if d <= 0:
                coloring[idx] = 1 - coloring[idx]
                current_count += d
                if current_count < best_count:
                    best_count = current_count
                    best_coloring = coloring.copy()
            else:
                # Accept with probability exp(-d/T)
                if np.random.random() < np.exp(-d / T):
                    coloring[idx] = 1 - coloring[idx]
                    current_count += d
                    if current_count < best_count:
                        best_count = current_count
                        best_coloring = coloring.copy()
        
        T *= cooling_rate
    
    # Phase 3: Greedy polish with 1-flip and 2-flip moves
    coloring = best_coloring.copy()
    current_count = best_count
    
    def greedy_polish(coloring, current_count, max_time):
        """Greedy polish using 1-flip and 2-flip moves."""
        nonlocal best_count, best_coloring
        improved = True
        while improved and time.time() < max_time:
            improved = False
            # Try 1-flips
            for idx in range(n):
                d = flip_delta_fast(coloring, idx)
                if d < 0:
                    coloring[idx] = 1 - coloring[idx]
                    current_count += d
                    improved = True
                    if current_count < best_count:
                        best_count = current_count
                        best_coloring = coloring.copy()
            if time.time() > max_time:
                break
            # Try 2-flips (sampled pairs for speed)
            if improved:
                # Sample some pairs
                for _ in range(200):
                    if time.time() > max_time:
                        break
                    i1 = np.random.randint(0, n)
                    i2 = np.random.randint(0, n)
                    if i1 == i2:
                        continue
                    d2 = flip_delta_two(coloring, i1, i2)
                    if d2 < 0:
                        coloring[i1] = 1 - coloring[i1]
                        coloring[i2] = 1 - coloring[i2]
                        current_count += d2
                        improved = True
                        if current_count < best_count:
                            best_count = current_count
                            best_coloring = coloring.copy()
        return current_count
    
    current_count = greedy_polish(coloring, current_count, deadline - 20)
    
    # Phase 4: Perturbation + polish restarts
    for restart in range(30):
        if time.time() > deadline - 5:
            break
        coloring = best_coloring.copy()
        current_count = best_count
        # Random perturbation: flip 2-4 random positions
        n_flips = np.random.randint(2, 5)
        for _ in range(n_flips):
            idx = np.random.randint(0, n)
            coloring[idx] = 1 - coloring[idx]
        current_count = count_mono(coloring)
        # Greedy polish
        current_count = greedy_polish(coloring, current_count, deadline - 5)
    
    return [int(c) for c in best_coloring]
# EVOLVE-BLOCK-END
