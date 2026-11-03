import numpy as np
import time
from scipy.spatial import cKDTree

N = 300

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin

    # Precompute pair indices for efficiency
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    num_pairs = len(i_idx)

    def compute_score(pts):
        """Compute the packing radius score."""
        tree = cKDTree(pts)
        # Query k nearest neighbors
        k = min(20, N-1)
        dists_k, idx_k = tree.query(pts, k=k+1)
        # Also need boundary distances
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        
        # Find minimum pair distance
        min_pair_dist = np.inf
        for i in range(N):
            for j in range(1, k+1):
                d = dists_k[i, j]
                if d > 1e-10:
                    min_pair_dist = min(min_pair_dist, d)
        
        min_bound = np.min(bound_dists)
        return min(min_pair_dist / 2.0, min_bound)

    def optimize(pts, alpha_init, k_init, k_end, cooling, time_budget,
                 stall_threshold=100, noise_scale_base=0.01):
        """Main optimization loop using soft-min gradient ascent with KD-tree."""
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        
        alpha = alpha_init
        k = k_init
        stall_count = 0
        end_time = start_time + time_budget
        
        while time.time() < end_time:
            # Compute distances using KD-tree for efficiency
            tree = cKDTree(pts)
            k_neighbors = min(20, N-1)
            dists_k, idx_k = tree.query(pts, k=k_neighbors+1)
            
            rad = np.linalg.norm(pts, axis=1)
            bound_dists = 1.0 - rad
            
            # Collect all relevant distances (pairs within k-nearest + boundary)
            pair_vals_list = []
            pair_i_list = []
            pair_j_list = []
            for i in range(N):
                for j in range(1, k_neighbors+1):
                    d = dists_k[i, j]
                    if d > 1e-10:
                        j_idx = idx_k[i, j]
                        # Avoid double counting
                        if i < j_idx:
                            pair_vals_list.append(d / 2.0)
                            pair_i_list.append(i)
                            pair_j_list.append(j_idx)
            
            pair_vals = np.array(pair_vals_list) if pair_vals_list else np.array([np.inf])
            pair_i_arr = np.array(pair_i_list) if pair_i_list else np.array([], dtype=int)
            pair_j_arr = np.array(pair_j_list) if pair_j_list else np.array([], dtype=int)
            
            all_vals = np.concatenate([pair_vals, bound_dists])
            current_min = np.min(all_vals)
            
            if current_min > best_score:
                best_score = current_min
                best_pts = pts.copy()
                stall_count = 0
            else:
                stall_count += 1
            
            # Compute soft-min weights
            shifted = all_vals - current_min
            weights = np.exp(-k * shifted)
            sum_w = np.sum(weights)
            if sum_w > 1e-15:
                weights /= sum_w
            else:
                weights = np.ones_like(all_vals) / len(all_vals)
            
            # Compute gradients
            grads = np.zeros_like(pts)
            
            # Pair contributions
            if len(pair_vals_list) > 0:
                pair_weights = weights[:len(pair_vals_list)]
                for idx, (i, j) in enumerate(zip(pair_i_arr, pair_j_arr)):
                    d_val = dists_k[i, np.where(idx_k[i] == j)[0][0]]
                    if d_val > 1e-10:
                        vec = pts[i] - pts[j]
                        vec_norm = vec / d_val
                        w = pair_weights[idx]
                        grads[i] += w * vec_norm
                        grads[j] -= w * vec_norm
            
            # Boundary contributions
            bound_weights = weights[len(pair_vals_list):]
            rad_safe = np.where(rad > 1e-10, rad, 1.0)
            bound_grads = -(pts / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
            grads += bound_grads
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grads)
            if grad_norm > 1e-10:
                grads /= grad_norm
            else:
                break
            
            # Update with adaptive step
            pts += alpha * grads
            
            # Project back inside circle
            rad_new = np.linalg.norm(pts, axis=1)
            mask_out = rad_new > 1.0
            if np.any(mask_out):
                pts[mask_out] /= rad_new[mask_out, np.newaxis]
            
            # Cool down
            alpha *= cooling
            
            # Increase k (sharpen the min)
            progress = min(1.0, (time.time() - start_time) / time_budget)
            k = k_init + (k_end - k_init) * (progress ** 1.5)
            
            # Random restart on stall
            if stall_count > stall_threshold:
                noise_scale = noise_scale_base * (1.0 + alpha * 10)
                pts += np.random.normal(0, noise_scale, size=pts.shape)
                rad_new = np.linalg.norm(pts, axis=1)
                mask_out = rad_new > 1.0
                if np.any(mask_out):
                    pts[mask_out] /= rad_new[mask_out, np.newaxis]
                stall_count = 0
                alpha = max(alpha, 0.005)
        
        return best_pts, best_score

    def local_search(pts, time_budget):
        """Simple local search: repeatedly push apart the closest pair."""
        pts = pts.copy()
        best_pts = pts.copy()
        best_score = compute_score(pts)
        end_time = start_time + time_budget
        
        while time.time() < end_time:
            tree = cKDTree(pts)
            k_neighbors = min(10, N-1)
            dists_k, idx_k = tree.query(pts, k=k_neighbors+1)
            
            rad = np.linalg.norm(pts, axis=1)
            bound_dists = 1.0 - rad
            
            # Find the minimum constraint
            min_pair_dist = np.inf
            min_pair_i = 0
            min_pair_j = 0
            for i in range(N):
                for j in range(1, k_neighbors+1):
                    d = dists_k[i, j]
                    if d > 1e-10 and d < min_pair_dist:
                        min_pair_dist = d
                        min_pair_i = i
                        min_pair_j = idx_k[i, j]
            
            min_bound_idx = np.argmin(bound_dists)
            min_bound_val = bound_dists[min_bound_idx]
            
            # Determine which constraint is most violated
            if min_pair_dist / 2.0 < min_bound_val:
                i, j = min_pair_i, min_pair_j
                vec = pts[i] - pts[j]
                norm = np.linalg.norm(vec)
                if norm > 1e-10:
                    vec /= norm
                    step = 0.002
                    pts[i] += step * vec
                    pts[j] -= step * vec
            else:
                i = min_bound_idx
                vec = pts[i]
                norm = np.linalg.norm(vec)
                if norm > 1e-10:
                    vec /= norm
                    step = 0.002
                    pts[i] -= step * vec
            
            # Project
            rad_new = np.linalg.norm(pts, axis=1)
            mask_out = rad_new > 1.0
            if np.any(mask_out):
                pts[mask_out] /= rad_new[mask_out, np.newaxis]
            
            # Update best
            current_score = compute_score(pts)
            if current_score > best_score:
                best_score = current_score
                best_pts = pts.copy()
        
        return best_pts, best_score

    def hex_lattice_init(spacing):
        """Generate hexagonal lattice points inside unit circle with proper selection."""
        y_step = spacing * np.sqrt(3) / 2
        max_j = int(1.0 / y_step) + 2
        max_i = int(1.0 / spacing) + 2
        
        pts = []
        for j in range(-max_j, max_j + 1):
            y = j * y_step
            offset = 0 if j % 2 == 0 else spacing / 2
            for i in range(-max_i, max_i + 1):
                x = i * spacing + offset
                if np.hypot(x, y) < 0.98:
                    pts.append([x, y])
        
        pts = np.array(pts)
        
        if len(pts) >= N:
            # Select N points using farthest-point sampling for good distribution
            selected = [np.random.choice(len(pts))]
            while len(selected) < N:
                dists = np.min([np.linalg.norm(pts - pts[s], axis=1) for s in selected], axis=0)
                next_idx = np.argmax(dists)
                selected.append(next_idx)
            pts = pts[selected]
        elif len(pts) < N:
            while len(pts) < N:
                p = np.random.uniform(-0.95, 0.95, 2)
                if np.linalg.norm(p) < 0.95:
                    pts = np.vstack([pts, p])
            pts = pts[:N]
        
        return pts

    def random_init():
        """Random initialization with rejection sampling."""
        pts = []
        while len(pts) < N:
            p = np.random.uniform(-0.98, 0.98, 2)
            if np.linalg.norm(p) < 0.98:
                pts.append(p)
        return np.array(pts)

    # Generate multiple initializations
    best_pts = None
    best_score = 0.0
    
    # Phase 1: Try hexagonal lattice with various spacings
    base_spacing = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    spacing_multipliers = [0.88, 0.93, 0.98, 1.03]
    
    for mult in spacing_multipliers:
        if time.time() - start_time > time_limit * 0.45:
            break
        
        spacing = base_spacing * mult
        pts = hex_lattice_init(spacing)
        
        time_budget = min(12.0, time_limit * 0.12)
        pts_opt, score = optimize(pts,
                                   alpha_init=0.015,
                                   k_init=8.0,
                                   k_end=800.0,
                                   cooling=0.998,
                                   time_budget=time_budget,
                                   stall_threshold=40,
                                   noise_scale_base=0.008)
        
        if score > best_score:
            best_score = score
            best_pts = pts_opt.copy()
    
    # Phase 2: Try random initializations
    if time.time() - start_time < time_limit * 0.65:
        for _ in range(2):
            if time.time() - start_time > time_limit * 0.65:
                break
            
            pts = random_init()
            time_budget = min(8.0, time_limit * 0.08)
            pts_opt, score = optimize(pts,
                                       alpha_init=0.01,
                                       k_init=3.0,
                                       k_end=300.0,
                                       cooling=0.997,
                                       time_budget=time_budget,
                                       stall_threshold=25,
                                       noise_scale_base=0.015)
            
            if score > best_score:
                best_score = score
                best_pts = pts_opt.copy()
    
    # Phase 3: Polish with gradient-based optimization (sharper)
    if best_pts is not None and time.time() - start_time < time_limit * 0.80:
        remaining_time = time_limit * 0.88 - time.time()
        if remaining_time > 3.0:
            pts_polished, score_polished = optimize(best_pts,
                                                     alpha_init=0.003,
                                                     k_init=1000.0,
                                                     k_end=5000.0,
                                                     cooling=0.994,
                                                     time_budget=min(remaining_time, 12.0),
                                                     stall_threshold=80,
                                                     noise_scale_base=0.0005)
            
            if score_polished > best_score:
                best_score = score_polished
                best_pts = pts_polished
    
    # Phase 4: Local search to escape local minima
    if best_pts is not None and time.time() - start_time < time_limit * 0.92:
        remaining_time = time_limit * 0.96 - time.time()
        if remaining_time > 3.0:
            pts_ls, score_ls = local_search(best_pts, min(remaining_time, 8.0))
            
            if score_ls > best_score:
                best_score = score_ls
                best_pts = pts_ls

    # Final verification
    if best_pts is None:
        best_pts = hex_lattice_init(base_spacing)
    
    # Ensure all points are inside the circle
    rad = np.linalg.norm(best_pts, axis=1)
    mask_out = rad > 1.0
    if np.any(mask_out):
        best_pts[mask_out] /= rad[mask_out, np.newaxis]
    
    return best_pts
