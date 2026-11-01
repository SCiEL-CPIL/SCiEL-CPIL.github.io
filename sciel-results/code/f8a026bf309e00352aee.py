import numpy as np
import time
from scipy.optimize import minimize

N = 50

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    timeout = 115  # Safety margin
    
    def _score(centers):
        """Compute the actual score (minimum radius) for a configuration."""
        c = centers
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist2, np.inf)
        min_dist = np.sqrt(np.min(dist2)) / 2.0
        
        abs_c = np.abs(c)
        max_coord = np.max(abs_c, axis=1)
        min_wall = 0.5 - np.max(max_coord)
        
        return min(min_dist, min_wall)
    
    def _smooth_min_distance(centers, beta):
        """
        Compute a smooth surrogate for the minimum pairwise distance using log-sum-exp.
        Returns (smooth_min_dist, gradient)
        """
        c = centers
        n = len(c)
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(dist2, 0)
        
        # Only consider upper triangle to avoid double counting
        iu = np.triu_indices(n, k=1)
        d2 = dist2[iu]
        
        if len(d2) == 0:
            return 0.0, np.zeros_like(c)
        
        # Soft minimum: -log(sum(exp(-d2/beta))) / beta
        # For numerical stability, use the log-sum-exp trick
        min_d2 = np.min(d2)
        exp_terms = np.exp(-(d2 - min_d2) / beta)
        log_sum = np.log(np.sum(exp_terms))
        
        # smooth_min_dist = sqrt(min_d2 + beta * log_sum) ... 
        # Actually, let's use a different approach:
        # We want to approximate min(d_ij) where d_ij = sqrt(d2_ij)
        # Use: f = -beta * log(sum(exp(-d_ij/beta)))
        # This is the soft-min of the distances
        
        # But computing d_ij = sqrt(d2_ij) for all pairs is expensive
        # Alternative: work with d2 and use soft-min of d2, then take sqrt
        # soft_min_d2 = min_d2 + beta * log(sum(exp(-(d2-min_d2)/beta)))
        # This approximates the minimum d2, not d.
        
        # Let's use the distance-based soft min:
        d = np.sqrt(d2)
        min_d = np.min(d)
        exp_d = np.exp(-(d - min_d) / beta)
        log_sum_d = np.log(np.sum(exp_d))
        soft_min_d = min_d + beta * log_sum_d
        
        # Gradient of soft_min_d with respect to centers
        # d(soft_min_d)/d(c_i) = sum_j w_ij * (c_i - c_j) / d_ij
        # where w_ij = exp(-(d_ij - min_d)/beta) / sum(exp(...))
        
        weights = exp_d / np.sum(exp_d)
        
        grad = np.zeros_like(c)
        for k in range(len(iu[0])):
            i, j = iu[0][k], iu[1][k]
            w = weights[k]
            diff_ij = c[i] - c[j]
            d_ij = d[k]
            if d_ij > 1e-10:
                grad[i] += w * diff_ij / d_ij
                grad[j] -= w * diff_ij / d_ij
        
        return soft_min_d, grad
    
    def _smooth_wall_distance(centers, beta):
        """
        Compute smooth surrogate for minimum wall distance.
        """
        c = centers
        abs_c = np.abs(c)
        max_coord = np.max(abs_c, axis=1)
        wall_dists = 0.5 - max_coord
        
        min_wall = np.min(wall_dists)
        exp_w = np.exp(-(wall_dists - min_wall) / beta)
        log_sum_w = np.log(np.sum(exp_w))
        soft_wall = min_wall + beta * log_sum_w
        
        # Gradient
        weights = exp_w / np.sum(exp_w)
        grad = np.zeros_like(c)
        for i in range(N):
            w = weights[i]
            # d(wall_dist_i)/d(c_i) = -sign(c_ik) for the k that achieves max
            for k in range(3):
                if abs(c[i][k]) == max_coord[i]:
                    grad[i][k] -= w * np.sign(c[i][k])
        
        return soft_wall, grad
    
    def _smooth_objective(centers, r, beta):
        """
        Smooth objective: penalize when soft_min_dist < 2r or soft_wall < r
        """
        soft_d, grad_d = _smooth_min_distance(centers, beta)
        soft_w, grad_w = _smooth_wall_distance(centers, beta)
        
        # Penalty terms
        pair_viol = max(0, 2*r - soft_d)
        wall_viol = max(0, r - soft_w)
        
        obj = pair_viol**2 + wall_viol**2
        
        # Gradient
        grad = np.zeros_like(centers)
        if pair_viol > 0:
            grad -= 2 * pair_viol * grad_d
        if wall_viol > 0:
            grad -= 2 * wall_viol * grad_w
        
        return obj, grad
    
    def _initial_grid():
        """Generate initial grid configuration."""
        m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
        return grid[:N].copy()
    
    def _refine_lloyd(centers, r, max_iter=100, tol=1e-8):
        """
        Lloyd-style refinement: move each center away from closest neighbors.
        """
        for it in range(max_iter):
            if time.time() - start_time > timeout - 5:
                break
                
            c = centers.copy()
            for i in range(N):
                # Find closest other center
                diff = c - c[i]
                dist2 = np.sum(diff * diff, axis=1)
                dist2[i] = np.inf
                j = np.argmin(dist2)
                d = np.sqrt(dist2[j])
                
                if d < 2 * r:
                    # Move i away from j
                    direction = (c[i] - c[j]) / d
                    c[i] += 0.1 * direction * (2 * r - d)
            
            # Project back into cube
            c = np.clip(c, -0.5 + r, 0.5 - r)
            
            centers = c
            
            # Check convergence
            if np.allclose(centers, c, atol=tol):
                break
        
        return centers
    
    def minimize_for_radius(r, initial_centers, beta_schedule=None):
        """
        Try to find a configuration for radius r using smooth optimization.
        """
        if beta_schedule is None:
            beta_schedule = [1.0, 0.5, 0.25, 0.1, 0.05, 0.02]
        
        x0 = initial_centers.flatten()
        bounds = [(-0.5 + r - 0.01, 0.5 - r + 0.01)] * (3 * N)
        
        best_x = x0.copy()
        best_obj = np.inf
        
        for beta in beta_schedule:
            if time.time() - start_time > timeout - 5:
                break
            
            def objective(x):
                c = x.reshape(N, 3)
                obj, _ = _smooth_objective(c, r, beta)
                return obj
            
            def gradient(x):
                c = x.reshape(N, 3)
                _, grad = _smooth_objective(c, r, beta)
                return grad.flatten()
            
            result = minimize(
                objective,
                best_x,
                jac=gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'disp': False, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_x = result.x.copy()
        
        # Final polish with tight tolerance
        if time.time() - start_time < timeout - 5:
            def final_objective(x):
                c = x.reshape(N, 3)
                obj, _ = _smooth_objective(c, r, 0.01)
                return obj
            
            def final_gradient(x):
                c = x.reshape(N, 3)
                _, grad = _smooth_objective(c, r, 0.01)
                return grad.flatten()
            
            result = minimize(
                final_objective,
                best_x,
                jac=final_gradient,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'disp': False, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            if result.fun < best_obj:
                best_obj = result.fun
                best_x = result.x.copy()
        
        best_centers = best_x.reshape(N, 3)
        
        # Check if feasible
        actual_score = _score(best_centers)
        success = actual_score >= r - 1e-8
        
        return success, best_centers, best_obj
    
    # Initial configuration
    current_centers = _initial_grid()
    
    # Binary search on radius
    r_low = 0.125
    r_high = 0.140
    
    best_centers = current_centers.copy()
    best_score = _score(best_centers)
    
    while time.time() - start_time < timeout and (r_high - r_low) > 1e-5:
        r_mid = (r_low + r_high) / 2.0
        
        success, new_centers, energy = minimize_for_radius(r_mid, current_centers)
        
        if success:
            r_low = r_mid
            current_centers = new_centers
            score = _score(new_centers)
            if score > best_score:
                best_score = score
                best_centers = new_centers
        else:
            r_high = r_mid
            # Update current_centers to the best found, even if not feasible
            current_centers = new_centers
        
        # Periodically try Lloyd refinement
        if int((r_high - r_low) * 100000) % 3 == 0:
            refined = _refine_lloyd(current_centers, r_low, max_iter=50)
            score = _score(refined)
            if score > best_score:
                best_score = score
                best_centers = refined
                current_centers = refined
    
    # Final refinement
    if time.time() - start_time < timeout - 5:
        final_refined = _refine_lloyd(best_centers, best_score, max_iter=200)
        final_score = _score(final_refined)
        if final_score > best_score:
            best_centers = final_refined
    
    return best_centers
# EVOLVE-BLOCK-END
