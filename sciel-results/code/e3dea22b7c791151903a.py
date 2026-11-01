import numpy as np
import time
from scipy.optimize import minimize

N = 300

def _hex_lattice_init(n):
    """Generate hexagonal lattice initialization for circle packing."""
    # Spacing s such that we get at least n points in unit circle
    # Approx area per point s^2 * sqrt(3)/2. Total area pi.
    # n * s^2 * sqrt(3)/2 <= pi -> s <= sqrt(2*pi / (n*sqrt(3)))
    s = np.sqrt(2 * np.pi / (n * np.sqrt(3))) * 0.95  # 5% margin
    
    # Generate grid
    k = int(1.0 / s) + 2
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * s
            y = j * s * np.sqrt(3.0) / 2.0
            r = np.hypot(x, y)
            if r <= 1.0:
                points.append((x, y, r))
                
    points = np.array(points)
    # Sort by distance to center
    points = points[points[:, 2].argsort()]
    
    # If we have more than N, take the first N
    if len(points) >= n:
        return points[:n, :2]
    else:
        # Fallback: scale down to fit
        r_max = np.max(points[:, 2])
        scale = 1.0 / r_max
        return points[:, :2] * scale

def _calc_score(x, idx_i, idx_j):
    """Calculate the true packing score: min(min pairwise distance/2, min boundary distance)."""
    diff = x[idx_i] - x[idx_j]
    d_ij = np.linalg.norm(diff, axis=1)
    min_d_ij = np.min(d_ij) / 2.0
    
    r_i = np.linalg.norm(x, axis=1)
    min_d_bound = np.min(1.0 - r_i)
    
    return np.min([min_d_ij, min_d_bound])

def _get_critical_pairs(x, threshold_ratio=1.05):
    """
    Identify pairs of circles that are close to the minimum distance,
    and circles close to the boundary.
    Returns:
        idx_i, idx_j: indices of critical pairs
        critical_boundary: indices of circles close to boundary
    """
    # Calculate all pairwise distances
    # To save memory and time, we can do this in chunks or just use broadcasting if N is small enough
    # N=300, 300x300 matrix is small (90k entries).
    
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    
    # Boundary distances
    r_i = np.linalg.norm(x, axis=1)
    d_bound = 1.0 - r_i
    
    # Find current score
    min_pair_dist = np.min(dists[np.triu_indices(N, 1)]) / 2.0
    min_bound_dist = np.min(d_bound)
    current_score = min(min_pair_dist, min_bound_dist)
    
    # Threshold for "critical"
    # We want pairs that are within threshold_ratio * current_score of the minimum
    # But we also need to consider if the boundary is the limiting factor.
    # If boundary is limiting, we care about circles near boundary.
    # If pairs are limiting, we care about pairs near min distance.
    
    # Let's just find the top K closest pairs and top K closest to boundary
    # This ensures we capture the active constraints.
    
    # Get indices of upper triangle
    i_idx, j_idx = np.triu_indices(N, 1)
    pair_dists = dists[i_idx, j_idx] / 2.0
    
    # Find the minimum pair distance
    min_pd = np.min(pair_dists)
    
    # Find pairs that are "close" to the minimum
    # Use a small epsilon to select the active set
    # If we select too many, optimization is slow. If too few, we might miss a constraint.
    # Let's select pairs within 1.02 * min_pd
    threshold = 1.02 * min_pd
    mask_pairs = pair_dists <= threshold
    crit_i = i_idx[mask_pairs]
    crit_j = j_idx[mask_pairs]
    
    # Find circles close to boundary
    min_bd = np.min(d_bound)
    threshold_b = 1.02 * min_bd
    mask_bound = d_bound <= threshold_b
    crit_b = np.where(mask_bound)[0]
    
    return crit_i, crit_j, crit_b

def construct():
    start_time = time.time()
    time_limit = 145.0  # Safety margin for 150s limit
    
    x0 = _hex_lattice_init(N)
    best_x = x0.copy()
    best_score = _calc_score(best_x, np.triu_indices(N, 1)[0], np.triu_indices(N, 1)[1])
    
    # Initial full optimization to get out of the lattice
    idx_i, idx_j = np.triu_indices(N, 1)
    
    def make_soft_min_objective(beta):
        def obj_and_grad(x_flat):
            x = x_flat.reshape(N, 2)
            diff = x[idx_i] - x[idx_j]
            d_ij = np.linalg.norm(diff, axis=1)
            d_ij_safe = np.where(d_ij < 1e-12, 1e-12, d_ij)
            
            r_i = np.linalg.norm(x, axis=1)
            d_bound = 1.0 - r_i
            
            y_pairs = d_ij / 2.0
            y_bound = d_bound
            all_y = np.concatenate([y_pairs, y_bound])
            y_min = np.min(all_y)
            
            exp_terms = np.exp(-beta * (all_y - y_min))
            sum_exp = np.sum(exp_terms)
            weights = exp_terms / sum_exp
            
            obj_val = y_min + (1.0 / beta) * np.log(sum_exp)
            
            grad = np.zeros_like(x)
            w_pairs = weights[:len(y_pairs)]
            terms = w_pairs[:, np.newaxis] * diff / (2.0 * d_ij_safe[:, np.newaxis])
            np.add.at(grad, idx_i, -terms)
            np.add.at(grad, idx_j, terms)
            
            w_bound = weights[len(y_pairs):]
            r_i_safe = np.where(r_i < 1e-12, 1e-12, r_i)
            grad_bound = -x / r_i_safe[:, np.newaxis]
            grad -= (w_bound[:, np.newaxis] * grad_bound)
            
            return obj_val, grad
        return obj_and_grad

    # Stage 1: Global optimization with moderate beta
    if time.time() - start_time < time_limit - 30:
        beta = 100
        obj_func = make_soft_min_objective(beta)
        try:
            res = minimize(
                obj_func, 
                best_x.flatten(), 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 100, 'gtol': 1e-7}
            )
            if res.x is not None and np.all(np.isfinite(res.x)):
                candidate_x = res.x.reshape(N, 2)
                score = _calc_score(candidate_x, idx_i, idx_j)
                if score > best_score:
                    best_x = candidate_x
                    best_score = score
        except Exception:
            pass

    # Stage 2: Iterative local refinement on critical sets
    # We will iterate, finding the critical constraints and optimizing only those.
    # This allows for much tighter tolerances and faster convergence per iteration.
    
    max_iter_local = 5
    for k in range(max_iter_local):
        if time.time() - start_time > time_limit - 10:
            break
            
        crit_i, crit_j, crit_b = _get_critical_pairs(best_x)
        
        # If no critical pairs found (shouldn't happen), break
        if len(crit_i) == 0:
            break
            
        # We need to optimize positions to maximize the minimum of these specific distances.
        # We can formulate this as a soft-min over just these critical distances.
        # This is a smaller problem.
        
        # Note: We must ensure that moving to improve these critical distances
        # does not violate other constraints. The soft-min approach with high beta
        # on the critical set approximates the max-min problem for the active set.
        # If a non-critical constraint becomes active, it will have a smaller distance
        # than the critical ones, so it would have been included in the next iteration's
        # critical set. However, during optimization, we might push a non-critical
        # distance below the current minimum.
        # To handle this, we can include a penalty for any distance < current_best_score.
        # Or, simpler: just optimize the critical set, and check the global score afterwards.
        # If global score drops, we revert.
        
        def make_local_objective(c_i, c_j, c_b, beta=500):
            def obj_and_grad(x_flat):
                x = x_flat.reshape(N, 2)
                
                # Critical pair distances
                diff = x[c_i] - x[c_j]
                d_ij = np.linalg.norm(diff, axis=1)
                d_ij_safe = np.where(d_ij < 1e-12, 1e-12, d_ij)
                y_pairs = d_ij / 2.0
                
                # Critical boundary distances
                r_i = np.linalg.norm(x, axis=1)
                d_bound = 1.0 - r_i
                y_bound = d_bound[c_b]
                
                all_y = np.concatenate([y_pairs, y_bound])
                if len(all_y) == 0:
                    return 0.0, np.zeros_like(x_flat)
                
                y_min = np.min(all_y)
                exp_terms = np.exp(-beta * (all_y - y_min))
                sum_exp = np.sum(exp_terms)
                weights = exp_terms / sum_exp
                
                obj_val = y_min + (1.0 / beta) * np.log(sum_exp)
                
                grad = np.zeros_like(x)
                
                # Gradient from pairs
                w_pairs = weights[:len(y_pairs)]
                terms = w_pairs[:, np.newaxis] * diff / (2.0 * d_ij_safe[:, np.newaxis])
                np.add.at(grad, c_i, -terms)
                np.add.at(grad, c_j, terms)
                
                # Gradient from boundary
                w_bound = weights[len(y_pairs):]
                # Only for critical boundary circles
                r_i_safe = np.where(r_i < 1e-12, 1e-12, r_i)
                grad_bound = -x / r_i_safe[:, np.newaxis]
                # Apply to specific indices
                grad[c_b] -= (w_bound[:, np.newaxis] * grad_bound[c_b])
                
                return obj_val, grad.flatten()
            return obj_and_grad

        local_obj = make_local_objective(crit_i, crit_j, crit_b, beta=500)
        
        try:
            res = minimize(
                local_obj, 
                best_x.flatten(), 
                method='L-BFGS-B', 
                jac=True, 
                options={'maxiter': 50, 'gtol': 1e-9}
            )
            
            if res.x is not None and np.all(np.isfinite(res.x)):
                candidate_x = res.x.reshape(N, 2)
                # Check if this candidate is valid and improved
                # We must ensure no new overlaps are created that are worse than best_score
                # The local optimization maximizes the min of the critical set.
                # It might violate non-critical constraints if they are close.
                # So we check the GLOBAL score.
                score = _calc_score(candidate_x, idx_i, idx_j)
                if score > best_score:
                    best_x = candidate_x
                    best_score = score
                else:
                    # If it didn't improve, we might be stuck.
                    # Try a smaller step or different beta?
                    # For now, just break if no improvement in a few iterations?
                    # Or continue to see if next critical set is different.
                    pass
        except Exception:
            pass

    return best_x
