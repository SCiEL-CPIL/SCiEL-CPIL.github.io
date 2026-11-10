import time
import random
import numpy as np

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit

    # Precompute all 4-term arithmetic progressions
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    total_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)  # shape (total_aps, 4)

    # For each index, store which APs contain it and the other 3 positions
    index_to_ap_indices = [[] for _ in range(N)]
    index_to_others = [[] for _ in range(N)]
    for ap_idx, (i1, i2, i3, i4) in enumerate(aps):
        for idx in (i1, i2, i3, i4):
            others = [x for x in (i1, i2, i3, i4) if x != idx]
            index_to_ap_indices[idx].append(ap_idx)
            index_to_others[idx].append(others)

    def count_mono(color):
        """Count monochromatic 4-APs using vectorized ops."""
        c = color[aps_arr]  # shape (total_aps, 4)
        # All 4 elements equal
        mono = np.all(c[:, 0] == c, axis=1)
        return int(np.sum(mono))

    def get_delta_vectorized(color, idx, new_val):
        """Compute delta for flipping color[idx] to new_val."""
        old_val = color[idx]
        if old_val == new_val:
            return 0
        others_list = index_to_others[idx]
        if not others_list:
            return 0
        # Stack all "others" triplets
        others_arr = np.array(others_list, dtype=np.int32)  # shape (k, 3)
        c_others = color[others_arr]  # shape (k, 3)
        # Check if all 3 others are same color
        all_same = np.all(c_others[:, 0] == c_others, axis=1)  # shape (k,)
        c_others_val = c_others[:, 0]  # shape (k,)
        
        delta = 0
        # Where all 3 others are same color:
        mask = all_same
        if np.any(mask):
            co = c_others_val[mask]
            # If old_val == co: currently mono, flipping breaks it -> delta -= 1
            # If old_val != co: currently non-mono, flipping makes it mono -> delta += 1
            delta += int(np.sum(co != old_val)) - int(np.sum(co == old_val))
        return delta

    def local_search_vectorized(color, max_iter=200, time_limit=5.0):
        """Vectorized hill climbing: try all flips simultaneously each iteration."""
        current_score = count_mono(color)
        t0 = time.time()
        for it in range(max_iter):
            if time.time() - t0 > time_limit:
                break
            improved = False
            # Try each index flip
            for idx in range(N):
                new_val = 1 - color[idx]
                delta = get_delta_vectorized(color, idx, new_val)
                if delta < 0:
                    color[idx] = new_val
                    current_score += delta
                    improved = True
            if not improved:
                break
        return current_score

    def sa_search(color, max_iter=20000, T_start=3.0, T_end=0.05, time_limit=10.0):
        """Simulated annealing with fast delta."""
        current_score = count_mono(color)
        best_score = current_score
        best_color = color.copy()
        T = T_start
        cooling = (T_end / T_start) ** (1.0 / max(max_iter, 1))
        t0 = time.time()
        rng_local = random.Random()
        
        for it in range(max_iter):
            if time.time() - t0 > time_limit:
                break
            idx = rng_local.randint(0, N - 1)
            new_val = 1 - color[idx]
            delta = get_delta_vectorized(color, idx, new_val)
            
            if delta < 0 or rng_local.random() < np.exp(-delta / max(T, 1e-10)):
                color[idx] = new_val
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_color = color.copy()
            
            T *= cooling
            if T < T_end:
                T = T_end
        
        return best_score, best_color

    def multi_start_local_search(color, n_starts=5, time_per_start=3.0):
        """Run local search from multiple perturbations of the given color."""
        best_score = count_mono(color)
        best_color = color.copy()
        t0 = time.time()
        for s in range(n_starts):
            if time.time() - t0 > n_starts * time_per_start:
                break
            perturbed = color.copy()
            # Random perturbation: flip ~10% of bits
            n_flips = N // 10
            flip_indices = random.sample(range(N), n_flips)
            for fi in flip_indices:
                perturbed[fi] = 1 - perturbed[fi]
            sc = local_search_vectorized(perturbed, max_iter=100, time_limit=time_per_start)
            if sc < best_score:
                best_score = sc
                best_color = perturbed.copy()
        return best_score, best_color

    # --- Initializations ---
    def random_init(seed=None):
        rng = random.Random(seed)
        return np.array([rng.randint(0, 1) for _ in range(N)], dtype=np.int8)

    def grid_init(pattern_type='checker', seed=None):
        """Create 2D grid-based initialization (16x16)."""
        grid = np.zeros((16, 16), dtype=np.int8)
        if pattern_type == 'checker':
            for i in range(16):
                for j in range(16):
                    grid[i, j] = (i + j) % 2
        elif pattern_type == 'row':
            for i in range(16):
                grid[i, :] = i % 2
        elif pattern_type == 'col':
            for j in range(16):
                grid[:, j] = j % 2
        elif pattern_type == 'diag':
            for i in range(16):
                for j in range(16):
                    grid[i, j] = (i - j) % 2
        elif pattern_type == 'block4':
            for i in range(16):
                for j in range(16):
                    grid[i, j] = ((i // 4) + (j // 4)) % 2
        elif pattern_type == 'block8':
            for i in range(16):
                for j in range(16):
                    grid[i, j] = ((i // 8) + (j // 8)) % 2
        elif pattern_type == 'mixed':
            # Mix of patterns
            for i in range(16):
                for j in range(16):
                    if i < 8:
                        grid[i, j] = (i + j) % 2
                    else:
                        grid[i, j] = (i // 4 + j // 4) % 2
        # Flatten row-major: position p maps to (p//16, p%16)
        flat = grid.flatten()
        return flat.astype(np.int8)

    def block_init(block_size):
        return np.array([(i // block_size) % 2 for i in range(N)], dtype=np.int8)

    def stripe_init(period):
        half = period // 2
        return np.array([(i % period) < half for i in range(N)]).astype(np.int8)

    # Generate diverse initial candidates
    candidates = []
    
    # Grid patterns
    for pt in ['checker', 'row', 'col', 'diag', 'block4', 'block8', 'mixed']:
        candidates.append(grid_init(pt))
    
    # Block patterns
    for bs in [2, 3, 4, 5, 6, 7, 8, 16]:
        candidates.append(block_init(bs))
    
    # Stripe patterns
    for p in [3, 5, 7, 9, 11, 13, 17]:
        candidates.append(stripe_init(p))
    
    # Random inits
    for s in range(30):
        candidates.append(random_init(seed=s))

    # Quick local search on all candidates
    scored = []
    for c in candidates:
        c = c.copy()
        sc = local_search_vectorized(c, max_iter=50, time_limit=2.0)
        scored.append((sc, c))
    
    scored.sort(key=lambda x: x[0])
    
    # Track best overall
    best_score = scored[0][0]
    best_color = scored[0][1].copy()
    
    # --- Phase 1: SA from top candidates ---
    top_k = min(10, len(scored))
    sa_results = []
    for i in range(top_k):
        if time.time() > deadline - 15.0:
            break
        sc, col = sa_search(scored[i][1].copy(), max_iter=15000, T_start=3.0, T_end=0.05, time_limit=8.0)
        sa_results.append((sc, col))
        if sc < best_score:
            best_score = sc
            best_color = col.copy()
    
    sa_results.sort(key=lambda x: x[0])
    
    # --- Phase 2: Multi-start local search from SA results ---
    for sc, col in sa_results[:5]:
        if time.time() > deadline - 10.0:
            break
        ms_sc, ms_col = multi_start_local_search(col, n_starts=8, time_per_start=2.0)
        if ms_sc < best_score:
            best_score = ms_sc
            best_color = ms_col.copy()
    
    # --- Phase 3: More SA runs with different parameters ---
    for trial in range(15):
        if time.time() > deadline - 5.0:
            break
        # Start from a random or perturbed best
        if trial % 2 == 0:
            start_col = best_color.copy()
            # Perturb
            n_flips = random.randint(5, 20)
            flip_indices = random.sample(range(N), n_flips)
            for fi in flip_indices:
                start_col[fi] = 1 - start_col[fi]
        else:
            start_col = random_init(seed=100 + trial)
        
        T_start = 2.0 + random.random() * 2.0
        sc, col = sa_search(start_col, max_iter=10000, T_start=T_start, T_end=0.03, time_limit=5.0)
        if sc < best_score:
            best_score = sc
            best_color = col.copy()
    
    # --- Phase 4: Final intensive local search ---
    if time.time() < deadline - 2.0:
        best_color = best_color.copy()
        best_score = local_search_vectorized(best_color, max_iter=500, time_limit=8.0)
    
    # --- Phase 5: One more SA polish ---
    if time.time() < deadline - 1.0:
        sc, col = sa_search(best_color.copy(), max_iter=5000, T_start=1.0, T_end=0.01, time_limit=3.0)
        if sc < best_score:
            best_score = sc
            best_color = col.copy()
        # Final local search
        best_score = local_search_vectorized(best_color, max_iter=200, time_limit=1.0)
    
    return [int(x) for x in best_color]
