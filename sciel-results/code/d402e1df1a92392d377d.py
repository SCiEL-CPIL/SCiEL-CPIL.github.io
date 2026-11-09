# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def kissing_number11() -> np.ndarray:
    """
    Constructs a collection of 11-dimensional integer points such that the minimum
    squared distance between any two points is >= the maximum squared norm of any point.
    
    Strategy:
    1. Start with E8⊕D3 base (252 points, all norm^2=8, min_dist^2=8)
    2. Try to find a better configuration by:
       a. Using a larger norm base (norm^2=12) if it allows more points
       b. Greedy expansion with efficient vectorized distance checks
       c. Multiple restarts with different candidate orderings
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s kill
    d = 11

    # ---------------------------------------------------------------
    # Helper: Build E8 minimal vectors scaled to norm^2 = 8
    # ---------------------------------------------------------------
    def build_e8_scaled():
        pts = []
        for i in range(8):
            for j in range(i + 1, 8):
                for si in [2, -2]:
                    for sj in [2, -2]:
                        v = np.zeros(8, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        pts.append(v)
        for signs in itertools.product([1, -1], repeat=8):
            if sum(1 for s in signs if s == -1) % 2 == 0:
                pts.append(np.array(signs, dtype=np.int64))
        return pts

    # ---------------------------------------------------------------
    # Helper: Build D3 minimal vectors scaled to norm^2 = 8
    # ---------------------------------------------------------------
    def build_d3_scaled():
        pts = []
        for i in range(3):
            for j in range(i + 1, 3):
                for si in [2, -2]:
                    for sj in [2, -2]:
                        v = np.zeros(3, dtype=np.int64)
                        v[i] = si
                        v[j] = sj
                        pts.append(v)
        return pts

    e8_pts = build_e8_scaled()  # 240 in R^8
    d3_pts = build_d3_scaled()  # 12 in R^3

    # ---------------------------------------------------------------
    # Build base 252-point config in R^11
    # ---------------------------------------------------------------
    base_points = []
    for v8 in e8_pts:
        v11 = np.zeros(11, dtype=np.int64)
        v11[:8] = v8
        base_points.append(v11)
    for v3 in d3_pts:
        v11 = np.zeros(11, dtype=np.int64)
        v11[8:] = v3
        base_points.append(v11)

    base_arr = np.array(base_points, dtype=np.int64)
    base_set = set(map(tuple, base_points))
    n_base = len(base_points)  # 252

    # ---------------------------------------------------------------
    # Generate all integer vectors in Z^11 with norm^2 <= max_norm_sq
    # ---------------------------------------------------------------
    def generate_all_candidates(max_norm_sq=8, dim=11):
        """Generate all nonzero integer vectors with 0 < ||v||^2 <= max_norm_sq."""
        results = []
        
        def gen(idx, current, norm_sq):
            if norm_sq > max_norm_sq:
                return
            if idx == dim:
                if norm_sq > 0:
                    results.append(current.copy())
                return
            max_val = int(np.sqrt(max_norm_sq - norm_sq))
            for val in range(-max_val, max_val + 1):
                new_norm = norm_sq + val * val
                if new_norm > max_norm_sq:
                    continue
                current[idx] = val
                gen(idx + 1, current, new_norm)
            current[idx] = 0

        gen(0, np.zeros(dim, dtype=np.int64), 0)
        return results

    # Pre-generate candidates for norm^2 <= 8
    candidates_8 = generate_all_candidates(8, 11)
    candidates_8 = [p for p in candidates_8 if tuple(p) not in base_set]
    candidates_8.sort(key=lambda p: int(np.sum(p ** 2)))

    # ---------------------------------------------------------------
    # Efficient greedy expansion function
    # ---------------------------------------------------------------
    def greedy_expand(initial_arr, initial_set, threshold, candidates, time_limit):
        """
        Greedily add candidates that maintain min_dist^2 >= threshold.
        Returns the expanded array.
        """
        current_arr = initial_arr.copy()
        current_set = initial_set.copy()
        k = len(current_arr)
        
        # Precompute norms for initial points
        norms_sq = np.sum(current_arr ** 2, axis=1)
        
        for p in candidates:
            if time.time() > time_limit:
                break
            
            p_norm_sq = int(np.sum(p ** 2))
            if p_norm_sq > threshold:
                continue  # Skip if norm exceeds threshold
            
            # Vectorized distance check
            dot_products = current_arr @ p
            dists_sq = p_norm_sq + norms_sq - 2 * dot_products
            min_dist = int(np.min(dists_sq))
            
            if min_dist >= threshold:
                current_arr = np.vstack([current_arr, p])
                current_set.add(tuple(p))
                norms_sq = np.vstack([norms_sq, [p_norm_sq]])
                k += 1
        
        return current_arr

    best_points = base_arr.copy()
    best_k = n_base

    # ---------------------------------------------------------------
    # Phase 1: Greedy expansion with norm^2 <= 8 candidates
    # ---------------------------------------------------------------
    if time.time() < deadline - 120:
        expanded = greedy_expand(base_arr, base_set, 8, candidates_8, deadline - 60)
        if len(expanded) > best_k:
            best_points = expanded
            best_k = len(expanded)

    # ---------------------------------------------------------------
    # Phase 2: Try a different approach - use a subset of E8 with higher norm
    # E8 has points with norm^2 = 4 (roots) and norm^2 = 8 (next shell)
    # Let's try to build a configuration with norm^2 = 12
    # ---------------------------------------------------------------
    if time.time() < deadline - 90:
        # Generate candidates with norm^2 <= 12
        candidates_12 = generate_all_candidates(12, 11)
        candidates_12 = [p for p in candidates_12 if tuple(p) not in base_set]
        candidates_12.sort(key=lambda p: int(np.sum(p ** 2)))
        
        # Try greedy expansion with threshold = 12
        # Start from a smaller base to allow higher norms
        # Use only the E8 part as base
        e8_base = []
        for v8 in e8_pts:
            v11 = np.zeros(11, dtype=np.int64)
            v11[:8] = v8
            e8_base.append(v11)
        e8_base_arr = np.array(e8_base, dtype=np.int64)
        e8_base_set = set(map(tuple, e8_base))
        
        expanded_12 = greedy_expand(e8_base_arr, e8_base_set, 12, candidates_12, deadline - 30)
        if len(expanded_12) > best_k:
            best_points = expanded_12
            best_k = len(expanded_12)

    # ---------------------------------------------------------------
    # Phase 3: Try multiple random orderings for greedy expansion
    # ---------------------------------------------------------------
    if time.time() < deadline - 60:
        import random
        random.seed(42)
        
        for trial in range(5):
            if time.time() > deadline - 30:
                break
            
            shuffled = candidates_8.copy()
            random.shuffle(shuffled)
            
            expanded = greedy_expand(base_arr, base_set, 8, shuffled, deadline - 15)
            if len(expanded) > best_k:
                best_points = expanded
                best_k = len(expanded)

    # ---------------------------------------------------------------
    # Final verification and return
    # ---------------------------------------------------------------
    points_arr = np.around(best_points).astype(np.int64)

    if len(points_arr) < 2 or points_arr.shape[1] != 11:
        points_arr = np.array([np.ones(11, dtype=np.int64), -np.ones(11, dtype=np.int64)])
        return points_arr

    # Check no zero vector
    zero_mask = np.all(points_arr == 0, axis=1)
    if np.any(zero_mask):
        points_arr = points_arr[~zero_mask]

    if len(points_arr) < 2:
        points_arr = np.array([np.ones(11, dtype=np.int64), -np.ones(11, dtype=np.int64)])
        return points_arr

    # Verify the constraint
    k_final = len(points_arr)
    min_dist_sq = float('inf')
    max_norm_sq = 0
    for i in range(k_final):
        n_sq = int(np.sum(points_arr[i] ** 2))
        if n_sq > max_norm_sq:
            max_norm_sq = n_sq
        for j in range(i + 1, k_final):
            diff = points_arr[i] - points_arr[j]
            d_sq = int(np.sum(diff ** 2))
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq

    if min_dist_sq < max_norm_sq:
        # Fallback to base 252
        points_arr = base_arr

    return points_arr

# EVOLVE-BLOCK-END
