# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools
import time

def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    # Precompute triangle indices
    tri_indices = np.array(list(itertools.combinations(range(n), 3)))
    i_idx = tri_indices[:, 0]
    j_idx = tri_indices[:, 1]
    k_idx = tri_indices[:, 2]
    
    def get_areas(points):
        """Compute areas of all triangles."""
        p_i = points[i_idx]
        p_j = points[j_idx]
        p_k = points[k_idx]
        # Area = 0.5 * |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)|
        # Vectorized:
        # cross product magnitude in 2D is x1*y2 - x2*y1
        # For triangle (p_i, p_j, p_k):
        # Area = 0.5 * |(p_j - p_i) x (p_k - p_i)|
        v1 = p_j - p_i
        v2 = p_k - p_i
        cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
        return 0.5 * np.abs(cross)

    def get_hull_area(points):
        """Compute area of convex hull."""
        try:
            hull = ConvexHull(points)
            return hull.volume
        except Exception:
            # If hull fails (e.g., collinear), return 0 to penalize
            return 0.0

    def true_score(points):
        """Compute the actual metric: A_min / A_hull."""
        areas = get_areas(points)
        a_min = np.min(areas)
        a_hull = get_hull_area(points)
        if a_hull == 0:
            return 0.0
        return a_min / a_hull

    def soft_objective(x, tau):
        """
        Objective function for optimization.
        Minimizes the negative soft-min of areas divided by hull area.
        """
        points = x.reshape((n, 2))
        areas = get_areas(points)
        a_hull = get_hull_area(points)
        if a_hull == 0 or not np.isfinite(a_hull):
            return 1e6  # Large penalty
        
        # Soft min: -tau * log(sum(exp(-areas/tau)))
        # To avoid overflow/underflow, use log-sum-exp trick
        scaled_areas = -areas / tau
        max_scaled = np.max(scaled_areas)
        log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled_areas - max_scaled)))
        soft_min = -tau * log_sum_exp
        
        score = soft_min / a_hull
        return -score  # Minimize negative score

    # Deadline guard
    start_time = time.time()
    timeout = 500  # Leave 100s margin for safety and evaluation overhead
    
    best_points = None
    best_score = -1.0
    
    # Initial random points
    rng = np.random.default_rng(seed=42)
    
    # Annealing schedule for tau
    # Start with large tau (smooth) and decrease to small tau (sharp min)
    tau_values = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005]
    
    # Multi-start strategy
    n_restarts = 5
    current_points = None
    
    for restart in range(n_restarts):
        if time.time() - start_time > timeout:
            break
            
        # Initialize
        if current_points is None:
            points = rng.random((n, 2))
        else:
            # Jitter previous best
            points = current_points + rng.normal(0, 0.05, (n, 2))
            # Keep within reasonable bounds, e.g., [0, 1]
            points = np.clip(points, 0, 1)
            
        x0 = points.flatten()
        
        # Annealing loop
        for tau in tau_values:
            if time.time() - start_time > timeout:
                break
                
            # Use Nelder-Mead for local optimization
            # Options: maxiter, xatol, fatol
            res = minimize(
                soft_objective, 
                x0, 
                args=(tau,), 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            
            x_opt = res.x
            points_opt = x_opt.reshape((n, 2))
            
            # Evaluate true score
            score = true_score(points_opt)
            
            if score > best_score:
                best_score = score
                best_points = points_opt.copy()
                current_points = points_opt.copy()
                
            # Update x0 for next tau
            x0 = x_opt

    if best_points is None:
        # Fallback to random
        best_points = rng.random((n, 2))
        
    return best_points

# EVOLVE-BLOCK-END
