import numpy as np
import time

N = 300

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We aim to finish by 145s.
    time_limit = 145.0
    
    # 1. Initialization: Hexagonal Lattice
    # Target spacing for N=300 in unit circle.
    # Approximate density: N ~ Area / (Area per hex) = pi / (s^2 * sqrt(3)/2)
    # s ~ sqrt(2*pi / (sqrt(3)*N))
    s_init = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
    
    # Generate a slightly denser lattice to have more points, then prune
    # Using 0.95 multiplier ensures we have enough points to select N from
    s_gen = s_init * 0.95
    y_step = s_gen * np.sqrt(3) / 2
    max_j = int(1.0 / y_step) + 2
    max_i = int(1.0 / s_gen) + 2
    
    points_list = []
    for j in range(-max_j, max_j + 1):
        y = j * y_step
        offset = 0 if j % 2 == 0 else s_gen / 2
        for i in range(-max_i, max_i + 1):
            x = i * s_gen + offset
            if np.hypot(x, y) < 1.0:
                points_list.append([x, y])
                
    points = np.array(points_list)
    
    if len(points) > N:
        # Keep points closest to the center to maximize packing density
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:N]
        points = points[idx]
    elif len(points) < N:
        # Fallback: fill with random points if lattice is too sparse (unlikely)
        while len(points) < N:
            p = np.random.uniform(-0.99, 0.99, 2)
            if np.linalg.norm(p) < 0.99:
                points = np.vstack([points, p])
        points = points[:N]

    points = points.copy()
    best_points = points.copy()
    
    # Precompute indices for pairwise distances
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    num_pairs = len(i_idx)
    
    def _score(pts):
        """Compute the packing score: min(pair_dist/2, 1-radius)."""
        d = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(d**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        pair_min = np.min(dists[i_idx, j_idx]) / 2.0
        rad = np.linalg.norm(pts, axis=1)
        bound_min = np.min(1.0 - rad)
        return min(pair_min, bound_min)
    
    best_score = _score(points)
    
    def _soft_min_weights(vals, k_soft):
        """
        Compute weights for the soft-min approximation.
        w_i = exp(-k * (v_i - v_min)) / sum(exp(-k * (v_j - v_min)))
        This focuses gradient updates on the constraints closest to the minimum.
        """
        current_min = np.min(vals)
        shifted = vals - current_min
        
        # Numerical stability: ignore terms that are far from the minimum
        # exp(-k * large_number) -> 0
        threshold = 50.0 / k_soft
        mask = shifted < threshold
        
        if np.any(mask):
            close_shifted = shifted[mask]
            # Log-sum-exp trick for stability
            max_close = np.max(close_shifted)
            if max_close > 0:
                exp_terms = np.exp(-k_soft * (close_shifted - max_close))
            else:
                exp_terms = np.exp(-k_soft * close_shifted)
            
            sum_exp = np.sum(exp_terms)
            if sum_exp > 0:
                weights = np.zeros_like(vals)
                weights[mask] = exp_terms / sum_exp
            else:
                weights = np.ones_like(vals) / len(vals)
        else:
            weights = np.ones_like(vals) / len(vals)
        
        return weights, current_min

    # Optimization parameters
    # Slower cooling allows for better exploration before convergence
    alpha = 0.02
    cooling = 0.99995  
    k_soft_start = 3.0
    k_soft_end = 150.0
    max_iter = 10000  # Cap iterations to avoid infinite loops, time guard is primary
    iter_count = 0
    stall_count = 0
    stall_threshold = 150
    
    k_soft = k_soft_start
    
    # Phase 1: Main Optimization (Soft-Min Gradient Ascent)
    # Allocate ~65% of time to this phase
    phase1_limit = time_limit * 0.65
    
    while time.time() - start_time < phase1_limit and iter_count < max_iter:
        # Compute pairwise distances efficiently
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        # Compute boundary distances
        rad = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - rad
        
        # All constraint values
        pair_vals = dists[i_idx, j_idx] / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        
        current_min = np.min(all_vals)
        
        if current_min > best_score:
            best_score = current_min
            best_points = points.copy()
            stall_count = 0
        else:
            stall_count += 1
        
        # Compute Soft-Min Weights
        weights, _ = _soft_min_weights(all_vals, k_soft)
        
        # Compute Gradients
        grads = np.zeros_like(points)
        
        # Pair contributions
        pair_weights = weights[:num_pairs]
        d_pair = dists[i_idx, j_idx]
        mask_nonzero = d_pair > 1e-9
        if np.any(mask_nonzero):
            vecs = np.zeros((num_pairs, 2))
            vecs[mask_nonzero] = (diff[i_idx[mask_nonzero], j_idx[mask_nonzero]] / d_pair[mask_nonzero, np.newaxis]) * pair_weights[mask_nonzero, np.newaxis]
            
            np.add.at(grads, i_idx, vecs)
            np.add.at(grads, j_idx, -vecs)
        
        # Boundary contributions
        bound_weights = weights[num_pairs:]
        rad_safe = np.where(rad > 1e-9, rad, 1.0)
        bound_grads = - (points / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads
        
        # Normalize gradient to prevent large steps
        grad_norm = np.linalg.norm(grads)
        if grad_norm > 1.0:
            grads /= grad_norm
        
        # Update points
        points += alpha * grads
        
        # Project back inside circle
        rad_new = np.linalg.norm(points, axis=1)
        mask = rad_new > 1.0
        if np.any(mask):
            points[mask] /= rad_new[mask, np.newaxis]
            
        # Cool down
        alpha *= cooling
        
        # Increase k_soft over time with smoother schedule
        progress = iter_count / max_iter
        k_soft = k_soft_start + (k_soft_end - k_soft_start) * (progress ** 1.5)
        
        # Random perturbation to escape local minima
        if stall_count > stall_threshold:
            noise_scale = alpha * 2.0
            points += np.random.normal(0, noise_scale, size=points.shape)
            rad_new = np.linalg.norm(points, axis=1)
            mask = rad_new > 1.0
            if np.any(mask):
                points[mask] /= rad_new[mask, np.newaxis]
            stall_count = 0
            alpha = max(alpha, 0.005)  # Don't let alpha get too small
        
        iter_count += 1

    # Phase 2: High-precision polish
    # Use the best points found so far
    points = best_points.copy()
    polish_alpha = 0.0005
    polish_k = 300.0
    polish_time_limit = time_limit * 0.95  # Leave 5% buffer for final checks
    
    # We use a very high k to make the soft-min approximation very sharp,
    # effectively optimizing the true min function.
    polish_iter = 0
    max_polish_iter = 5000
    
    while time.time() - start_time < polish_time_limit and polish_iter < max_polish_iter:
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        
        rad = np.linalg.norm(points, axis=1)
        bound_dists = 1.0 - rad
        
        pair_vals = dists[i_idx, j_idx] / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        
        current_min = np.min(all_vals)
        
        if current_min > best_score:
            best_score = current_min
            best_points = points.copy()
        
        # Very sharp soft-min for polish
        weights, _ = _soft_min_weights(all_vals, polish_k)
        
        grads = np.zeros_like(points)
        
        pair_weights = weights[:num_pairs]
        d_pair = dists[i_idx, j_idx]
        mask_nonzero = d_pair > 1e-9
        if np.any(mask_nonzero):
            vecs = np.zeros((num_pairs, 2))
            vecs[mask_nonzero] = (diff[i_idx[mask_nonzero], j_idx[mask_nonzero]] / d_pair[mask_nonzero, np.newaxis]) * pair_weights[mask_nonzero, np.newaxis]
            
            np.add.at(grads, i_idx, vecs)
            np.add.at(grads, j_idx, -vecs)
        
        bound_weights = weights[num_pairs:]
        rad_safe = np.where(rad > 1e-9, rad, 1.0)
        bound_grads = - (points / rad_safe[:, np.newaxis]) * bound_weights[:, np.newaxis]
        grads += bound_grads
        
        grad_norm = np.linalg.norm(grads)
        if grad_norm > 1.0:
            grads /= grad_norm
            
        points += polish_alpha * grads
        
        rad_new = np.linalg.norm(points, axis=1)
        mask = rad_new > 1.0
        if np.any(mask):
            points[mask] /= rad_new[mask, np.newaxis]
            
        polish_iter += 1

    # Final safety check: ensure all points are strictly inside
    rad_final = np.linalg.norm(best_points, axis=1)
    mask_out = rad_final > 1.0
    if np.any(mask_out):
        best_points[mask_out] /= rad_final[mask_out, np.newaxis]
        
    # Re-verify score after projection
    final_score = _score(best_points)
    if final_score > best_score:
        best_score = final_score
    else:
        # If projection reduced score, we might have moved a point that was critical.
        # But usually projection only happens if a point escaped, which shouldn't happen
        # if we project every step. If it did, the previous best_points was safer.
        # However, the evaluator requires valid points. If best_points was invalid, we must fix it.
        # If best_points was valid, projection is a no-op or tiny fix.
        pass

    return best_points
