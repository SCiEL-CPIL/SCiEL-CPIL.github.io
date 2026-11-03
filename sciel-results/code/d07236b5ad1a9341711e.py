import numpy as np
import time
from scipy.optimize import least_squares

N = 300

def construct():
    start_time = time.time()
    time_limit = 115.0  # Safety margin

    # 1. Initialization: Hexagonal Lattice
    s_init = np.sqrt(2 * np.pi / (np.sqrt(3) * N))
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
        dists = np.linalg.norm(points, axis=1)
        idx = np.argsort(dists)[:N]
        points = points[idx]
    elif len(points) < N:
        while len(points) < N:
            p = np.random.uniform(-1, 1, 2)
            if np.linalg.norm(p) < 1.0:
                points = np.vstack([points, p])
        points = points[:N]

    points = points.copy()
    best_points = points.copy()
    best_score = 0.0

    # Precompute indices for pairwise distances
    i_idx, j_idx = np.where(np.triu(np.ones((N, N), dtype=bool), k=1))
    num_pairs = len(i_idx)

    def compute_score(pts):
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        rad = np.linalg.norm(pts, axis=1)
        bound_dists = 1.0 - rad
        pair_vals = dists[i_idx, j_idx] / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        return np.min(all_vals)

    # Phase 1: Gradient-based soft-min optimization
    alpha = 0.02
    cooling = 0.9999
    k_soft_start = 5.0
    k_soft_end = 200.0
    max_iter = 5000
    iter_count = 0
    stall_count = 0
    stall_threshold = 100
    k_soft = k_soft_start

    while time.time() - start_time < time_limit * 0.6 and iter_count < max_iter:
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
            stall_count = 0
        else:
            stall_count += 1

        # Soft-min weights
        shifted_vals = all_vals - current_min
        max_shift = np.max(shifted_vals)
        if max_shift > 50.0:
            mask_close = shifted_vals < 50.0
            weights = np.zeros_like(all_vals)
            if np.any(mask_close):
                close_vals = shifted_vals[mask_close]
                close_weights = np.exp(-k_soft * close_vals)
                weights[mask_close] = close_weights
                sum_weights = np.sum(weights)
                if sum_weights > 0:
                    weights /= sum_weights
                else:
                    weights = np.ones_like(all_vals) / len(all_vals)
            else:
                weights = np.ones_like(all_vals) / len(all_vals)
        else:
            weights = np.exp(-k_soft * shifted_vals)
            sum_weights = np.sum(weights)
            if sum_weights > 0:
                weights /= sum_weights
            else:
                weights = np.ones_like(all_vals) / len(all_vals)

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

        points += alpha * grads

        rad_new = np.linalg.norm(points, axis=1)
        mask = rad_new > 1.0
        if np.any(mask):
            points[mask] /= rad_new[mask, np.newaxis]

        alpha *= cooling
        progress = iter_count / max_iter
        k_soft = k_soft_start + (k_soft_end - k_soft_start) * (progress ** 2)

        if stall_count > stall_threshold:
            noise_scale = alpha * 3.0
            points += np.random.normal(0, noise_scale, size=points.shape)
            rad_new = np.linalg.norm(points, axis=1)
            mask = rad_new > 1.0
            if np.any(mask):
                points[mask] /= rad_new[mask, np.newaxis]
            stall_count = 0
            alpha *= 0.8

        iter_count += 1

    # Phase 2: scipy.optimize.least_squares refinement
    # We formulate the problem as minimizing the "deficit" of constraints.
    # We want to maximize r such that d_ij >= 2r and 1 - |c_i| >= r.
    # Equivalent to minimizing max(0, 2r - d_ij) and max(0, r - (1-|c_i|)).
    # However, least_squares minimizes sum(residuals^2).
    # A common trick for packing is to minimize the sum of squared violations
    # of the condition d_ij >= 2*r_target and 1-|c_i| >= r_target.
    # But r is what we want to find.
    # Alternative: Minimize a function that pushes points apart and inwards.
    # Let's use a surrogate: Minimize sum ( (2*r_est - d_ij)_+^2 + (r_est - (1-|c_i|))_+^2 )
    # But r_est is unknown.
    # Better approach for least_squares:
    # Minimize residuals defined as:
    # For each pair (i,j): residual = max(0, 2*r_current - d_ij)
    # For each point i: residual = max(0, r_current - (1 - |c_i|))
    # This drives the configuration towards satisfying the constraints for r_current.
    # As we iterate, we can increase r_current slightly if all constraints are satisfied.
    
    # However, least_squares is a local solver. It works best when close to the solution.
    # We will use it to polish the result from Phase 1.
    
    points = best_points.copy()
    current_r = best_score
    
    # Define residuals for least_squares
    # We flatten the points to a 1D array of size 2*N
    def residuals(x_flat):
        x = x_flat.reshape(N, 2)
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        rad = np.linalg.norm(x, axis=1)
        bound_dists = 1.0 - rad
        
        pair_vals = dists[i_idx, j_idx] / 2.0
        all_vals = np.concatenate([pair_vals, bound_dists])
        
        # We want to push the minimum value up.
        # Residuals are the "slack" below the current target radius.
        # If a constraint is violated (val < current_r), we penalize it.
        # To make it smooth, we use a soft penalty.
        # residual = max(0, current_r - val)
        # But max(0, .) is not differentiable.
        # Use a smooth approximation: softplus(current_r - val) * beta
        # Or simply: if we are very close, we can use the gradient of the min directly.
        
        # Let's try a different formulation:
        # Minimize sum of (current_r - val)^2 for all val < current_r + epsilon
        # This effectively tries to raise the lowest values.
        
        # A simpler and more robust approach for least_squares in packing:
        # Minimize the sum of squared distances between points, weighted by inverse distance?
        # No, let's stick to the constraint violation idea with a smooth hinge.
        
        # Let t = current_r.
        # Residuals for pairs: max(0, t - d_ij/2)
        # Residuals for bounds: max(0, t - (1 - |c_i|))
        
        # To ensure differentiability, we use:
        # r_pair = (t - d_ij/2) if d_ij/2 < t else 0
        # This is kinked at d_ij/2 = t.
        # We can smooth it with: r_pair = 0.5 * (t - d_ij/2 + sqrt((t - d_ij/2)^2 + eps))
        
        eps = 1e-6
        pair_diffs = t - pair_vals
        pair_res = 0.5 * (pair_diffs + np.sqrt(pair_diffs**2 + eps))
        
        bound_diffs = t - bound_dists
        bound_res = 0.5 * (bound_diffs + np.sqrt(bound_diffs**2 + eps))
        
        return np.concatenate([pair_res, bound_res])
    
    # We need to be careful: if current_r is already the max possible, residuals are all 0 (or close).
    # We want to increase current_r.
    # Strategy:
    # 1. Run least_squares to minimize residuals for current_r. This will fix any violations.
    # 2. Once residuals are small, increase current_r slightly and repeat.
    
    polish_start_time = time.time()
    polish_time_limit = time_limit * 0.95
    
    # Initial r for polishing
    r_target = best_score * 0.999  # Start slightly lower to ensure feasibility
    
    for _ in range(50):  # Max 50 polish steps
        if time.time() - polish_start_time > polish_time_limit - 10:
            break
            
        x0 = points.flatten()
        
        try:
            res = least_squares(residuals, x0, args=(r_target,), method='trf', 
                                bounds=([-1.0]*2*N, [1.0]*2*N), 
                                max_nfev=1000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            
            new_points = res.x.reshape(N, 2)
            
            # Project back inside
            rad_new = np.linalg.norm(new_points, axis=1)
            mask = rad_new > 1.0
            if np.any(mask):
                new_points[mask] /= rad_new[mask, np.newaxis]
                
            # Check if this configuration is valid for r_target
            score_new = compute_score(new_points)
            
            if score_new >= r_target:
                # We can increase r_target
                points = new_points
                # Increase r_target by a small amount
                r_target *= 1.0001
                if score_new > best_score:
                    best_score = score_new
                    best_points = new_points.copy()
            else:
                # Configuration not good enough, maybe try a smaller step or break
                # If we can't even satisfy current r_target, we might be stuck
                # Let's try to recover to best_points
                points = best_points.copy()
                # Try a smaller increase next time or break
                # For now, just break to avoid wasting time
                break
                
        except Exception as e:
            break

    return best_points
