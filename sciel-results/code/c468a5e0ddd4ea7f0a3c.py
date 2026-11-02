import numpy as np
import time
import sys
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

def construct():
    """
    Constructs 130 points on the unit sphere to minimize the covering radius.
    Combines L-BFGS-B with analytical soft-min/soft-max gradients and Lloyd's refinement.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    # 1. Initial configuration: Fibonacci spiral
    def fibonacci_spiral(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        pts = np.column_stack([r * np.cos(phi), 
                               r * np.sin(phi), 
                               z])
        return pts / np.linalg.norm(pts, axis=1, keepdims=True)
    
    points_init = fibonacci_spiral(N)
    
    # 2. Generate test grids
    # Coarse grid for optimization
    M_opt = 4000
    grid_opt = fibonacci_spiral(M_opt)
    
    # Fine grid for final evaluation / refinement
    # Increased to 20000 for better approximation of the true max
    M_fine = 20000
    grid_fine = fibonacci_spiral(M_fine)
    
    best_points = points_init.copy()
    best_score = float('inf')
    
    # Helper: Compute exact covering radius from convex hull facets
    def compute_covering_radius(points):
        try:
            hull = ConvexHull(points)
        except Exception:
            return np.pi / 2.0 # Fallback
        
        max_radius = 0.0
        for simplex in hull.simplices:
            verts = points[simplex]
            a, b, c = verts[0], verts[1], verts[2]
            normal = np.cross(b - a, c - a)
            norm_val = np.linalg.norm(normal)
            
            if norm_val < 1e-12:
                continue
                
            normal = normal / norm_val
            
            # Ensure normal points outward
            centroid = np.mean(verts, axis=0)
            if np.dot(normal, centroid) < 0:
                normal = -normal
            
            # The covering radius for this facet is the max angle from normal to vertices
            dots = np.clip(np.dot(verts, normal), -1.0, 1.0)
            angles = np.arccos(dots)
            facet_radius = np.max(angles)
            max_radius = max(max_radius, facet_radius)
        
        return max_radius

    # Objective and Gradient using Soft-Min / Soft-Max
    def objective_and_grad(flat_points, grid, sigma_min=0.05, sigma_max=0.05):
        pts = flat_points.reshape(N, 3)
        
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        pts = pts / norms
        
        # Compute dots (M, N)
        dots = grid @ pts.T
        dots = np.clip(dots, -1.0, 1.0)
        angles = np.arccos(dots)
        
        # Soft-Min over points for each grid point
        # w_ij = exp(-angle_ij / sigma_min) / sum_k exp(-angle_ik / sigma_min)
        neg_scaled = -angles / sigma_min
        max_neg = np.max(neg_scaled, axis=1, keepdims=True)
        exp_neg = np.exp(neg_scaled - max_neg)
        weights = exp_neg / np.sum(exp_neg, axis=1, keepdims=True)
        
        # Soft-min value
        soft_min_vals = np.sum(weights * angles, axis=1)
        
        # Soft-Max over grid points
        # v_i = soft_min_vals[i]
        # max_val = sigma_max * (max(v/sigma_max) + log(sum(exp((v-max)/sigma_max))))
        scaled_max = soft_min_vals / sigma_max
        max_scaled = np.max(scaled_max)
        exp_max = np.exp(scaled_max - max_scaled)
        max_weights = exp_max / np.sum(exp_max)
        
        soft_max_val = sigma_max * (max_scaled + np.log(np.sum(exp_max)))
        
        # Gradient Calculation
        # d(soft_min_i)/d(p_j) = sum_k w_ik * d(angle_ik)/d(p_j)
        # d(angle_ik)/d(p_j) is non-zero only if j==k
        # d(arccos(g.p))/dp = -g / sqrt(1-(g.p)^2) - (g.p)/sqrt(1-(g.p)^2) * p  (projected to tangent)
        # Let denom = sqrt(1 - dots^2)
        # grad_angle_ij = -g_i/denom_ij + dots_ij/denom_ij * p_j
        
        # We need d(soft_max)/d(p_j) = sum_i max_weights_i * d(soft_min_i)/d(p_j)
        # d(soft_min_i)/d(p_j) = w_ij * grad_angle_ij
        
        # So d(soft_max)/d(p_j) = sum_i (max_weights_i * w_ij) * grad_angle_ij
        
        # Let coeff_ij = max_weights_i * w_ij
        coeff = max_weights[:, np.newaxis] * weights
        
        denom = np.sqrt(np.maximum(1.0 - dots * dots, 1e-18))
        
        # Term 1: -sum_i coeff_ij * g_i / denom_ij
        alpha = coeff / denom
        term1 = -(alpha.T @ grid)
        
        # Term 2: sum_i coeff_ij * dots_ij / denom_ij * p_j
        # = sum_i alpha_ij * dots_ij * p_j
        # = (sum_i alpha_ij * dots_ij) * p_j
        beta = alpha * dots
        beta_sum = np.sum(beta, axis=0)
        term2 = beta_sum[:, np.newaxis] * pts
        
        grad = term1 + term2
        
        return soft_max_val, grad

    def objective(flat_points):
        val, _ = objective_and_grad(flat_points, grid_opt)
        return val

    def gradient(flat_points):
        _, grad = objective_and_grad(flat_points, grid_opt)
        return grad.flatten()

    # 3. Optimization Loop with multiple restarts/phases
    current_points = points_init.copy()
    
    # Phase 1: L-BFGS-B with coarse grid
    x0 = current_points.flatten()
    
    # We might do a few rounds of optimization, increasing sigma or switching grids
    # Start with larger sigma for stability, then decrease
    
    sigmas = [0.2, 0.1, 0.05, 0.02]
    
    for sigma in sigmas:
        if time.time() > deadline - 10.0:
            break
            
        # Redefine objective with current sigma
        def obj_wrapper(flat, s=sigma):
            val, _ = objective_and_grad(flat, grid_opt, sigma_min=s, sigma_max=s)
            return val
            
        def grad_wrapper(flat, s=sigma):
            _, grad = objective_and_grad(flat, grid_opt, sigma_min=s, sigma_max=s)
            return grad.flatten()

        try:
            res = minimize(obj_wrapper, x0, method='L-BFGS-B', jac=grad_wrapper, 
                           options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6, 'disp': False})
            
            if res.success or res.fun < best_score:
                current_points = res.x.reshape(N, 3)
                current_points = current_points / np.linalg.norm(current_points, axis=1, keepdims=True)
                x0 = current_points.flatten()
                
                # Update best based on exact hull radius if it looks good
                # But hull radius is expensive, so we approximate with fine grid max dist
                dots_fine = grid_fine @ current_points.T
                dots_fine = np.clip(dots_fine, -1.0, 1.0)
                dists_fine = np.arccos(dots_fine)
                min_dists_fine = np.min(dists_fine, axis=1)
                approx_radius = np.max(min_dists_fine)
                
                if approx_radius < best_score:
                    best_score = approx_radius
                    best_points = current_points.copy()
                    
        except Exception as e:
            pass

    # Phase 2: Lloyd's Algorithm for refinement
    # Lloyd's moves points to the centroid of their Voronoi cell (on the sphere)
    # This is very effective for reducing covering radius
    
    def lloyd_iteration(points, grid, n_iters=50):
        pts = points.copy()
        for _ in range(n_iters):
            if time.time() > deadline - 5.0:
                break
                
            dots = grid @ pts.T
            dots = np.clip(dots, -1.0, 1.0)
            angles = np.arccos(dots)
            nearest = np.argmin(angles, axis=1)
            
            new_pts = np.zeros_like(pts)
            
            # Vectorized sum
            # Use bincount or simple loop since N is small
            for j in range(N):
                mask = (nearest == j)
                count = np.sum(mask)
                if count > 0:
                    new_pts[j] = np.sum(grid[mask], axis=0) / count
                else:
                    new_pts[j] = pts[j]
            
            # Normalize
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            new_pts = new_pts / norms
            
            # Check convergence
            diff = np.linalg.norm(new_pts - pts, axis=1)
            pts = new_pts
            
            if np.max(diff) < 1e-8:
                break
                
        return pts

    # Run Lloyd's on fine grid
    # Reduced iterations to save time for the more effective L-BFGS polish, 
    # as Lloyd's converges slowly in the final stages.
    current_points = lloyd_iteration(current_points, grid_fine, n_iters=50)
    
    # Evaluate Lloyd's result
    dots_fine = grid_fine @ current_points.T
    dots_fine = np.clip(dots_fine, -1.0, 1.0)
    dists_fine = np.arccos(dots_fine)
    min_dists_fine = np.min(dists_fine, axis=1)
    approx_radius = np.max(min_dists_fine)
    
    if approx_radius < best_score:
        best_score = approx_radius
        best_points = current_points.copy()

    # Phase 3: Final Polish with L-BFGS on fine grid if time permits
    if time.time() < deadline - 10.0:
        x0 = best_points.flatten()
        
        # Use smaller sigma to better approximate the hard max
        sigma_final = 0.005
        
        def obj_fine(flat):
            val, _ = objective_and_grad(flat, grid_fine, sigma_min=sigma_final, sigma_max=sigma_final)
            return val
            
        def grad_fine(flat):
            _, grad = objective_and_grad(flat, grid_fine, sigma_min=sigma_final, sigma_max=sigma_final)
            return grad.flatten()

        try:
            res = minimize(obj_fine, x0, method='L-BFGS-B', jac=grad_fine, 
                           options={'maxiter': 40, 'ftol': 1e-12, 'gtol': 1e-10, 'disp': False})
            
            if res.success or res.fun < best_score:
                final_points = res.x.reshape(N, 3)
                final_points = final_points / np.linalg.norm(final_points, axis=1, keepdims=True)
                
                # Verify with grid
                dots_final = grid_fine @ final_points.T
                dots_final = np.clip(dots_final, -1.0, 1.0)
                dists_final = np.arccos(dots_final)
                min_dists_final = np.min(dists_final, axis=1)
                final_radius = np.max(min_dists_final)
                
                if final_radius < best_score:
                    best_score = final_radius
                    best_points = final_points
        except Exception:
            pass

    # Final sanity check: ensure points are on sphere and valid
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    # One last Lloyd step on a very fine grid if time is left? 
    # No, just return best_points.
    
    return best_points
