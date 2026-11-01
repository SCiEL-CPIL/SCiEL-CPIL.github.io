"""Optimized program for covering the sphere with 130 equal caps."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
import time


def _spherical_grid(n_points=20000):
    """Generate a dense spherical grid using Fibonacci spiral."""
    i = np.arange(n_points) + 0.5
    z = 1.0 - 2.0 * i / n_points
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _covering_radius_exact(points, grid=None):
    """Compute the exact covering radius in radians using a dense grid."""
    if grid is None:
        grid = _spherical_grid()
    
    # Find nearest neighbor for each grid point
    tree = cKDTree(points)
    dists, _ = tree.query(grid)
    
    # Convert Euclidean distances to angular distances
    max_d = np.max(dists)
    if max_d >= 2.0:
        return np.pi
    return 2.0 * np.arcsin(np.clip(max_d / 2.0, -1.0, 1.0))


def _smooth_objective(x, grid, temperature=0.1):
    """
    Smooth, differentiable objective function using soft-min of squared distances.
    This creates a smooth landscape for gradient-based optimization.
    """
    pts = x.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    pts = pts / norms
    
    # Compute squared Euclidean distances from each grid point to each sphere point
    # grid shape: (M, 3), pts shape: (N, 3)
    # dist_sq[i,j] = ||grid[i] - pts[j]||^2 = 2 - 2 * dot(grid[i], pts[j])
    dots = grid @ pts.T  # (M, N)
    dist_sq = 2.0 - 2.0 * dots  # (M, N)
    
    # Use soft-min (log-sum-exp trick) to get a smooth approximation of max distance
    # For each grid point, find the nearest point (min dist_sq)
    # soft_min_j(dist_sq[i,j]) = -log(sum_j exp(-dist_sq[i,j] / temperature)) * temperature
    # This is smooth and differentiable
    
    # Clip to avoid numerical issues
    dist_sq = np.maximum(dist_sq, 0.0)
    
    # Soft min over points for each grid point
    scaled = -dist_sq / temperature
    # Use log-sum-exp for numerical stability
    max_scaled = np.max(scaled, axis=1, keepdims=True)
    log_sum_exp = max_scaled.squeeze() + np.log(np.sum(np.exp(scaled - max_scaled), axis=1))
    soft_min_dist_sq = -temperature * log_sum_exp
    
    # The covering radius is the max over grid points of the nearest distance
    # We want to minimize the max, so we can use a smooth max approximation
    # max_i(soft_min_dist_sq[i]) ≈ -log(sum_i exp(-soft_min_dist_sq[i] / temp2)) * temp2
    temp2 = temperature * 0.5
    scaled2 = -soft_min_dist_sq / temp2
    max_scaled2 = np.max(scaled2)
    log_sum_exp2 = max_scaled2 + np.log(np.sum(np.exp(scaled2 - max_scaled2)))
    smooth_max_dist_sq = -temp2 * log_sum_exp2
    
    # Convert squared distance to angular distance
    # d = sqrt(dist_sq), angle = 2 * arcsin(d/2)
    d = np.sqrt(np.maximum(smooth_max_dist_sq, 0.0))
    if d >= 2.0:
        return np.pi
    return 2.0 * np.arcsin(np.clip(d / 2.0, -1.0, 1.0))


def _initial_points():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _local_refine(points, grid, max_iter=200, step_size=0.01):
    """
    Local refinement using subgradient method on the exact covering radius.
    For each grid point, identify which sphere point is nearest.
    Push that sphere point away from the grid point to reduce the maximum distance.
    """
    pts = points.copy()
    current_radius = _covering_radius_exact(pts, grid)
    
    for iteration in range(max_iter):
        # Find nearest sphere point for each grid point
        tree = cKDTree(pts)
        dists, indices = tree.query(grid)
        
        # Find the grid point with maximum distance (worst case)
        max_idx = np.argmax(dists)
        worst_grid_point = grid[max_idx]
        nearest_sphere_idx = indices[max_idx]
        current_dist = dists[max_idx]
        
        if current_dist < 1e-8:
            break
        
        # Compute direction to move the sphere point
        # We want to move pts[nearest_sphere_idx] away from worst_grid_point
        # on the sphere
        move_dir = pts[nearest_sphere_idx] - worst_grid_point
        move_norm = np.linalg.norm(move_dir)
        if move_norm < 1e-10:
            # If they're nearly coincident, move in a random direction
            move_dir = np.random.randn(3)
            move_norm = np.linalg.norm(move_dir)
        move_dir = move_dir / move_norm
        
        # Move the sphere point
        pts[nearest_sphere_idx] += step_size * move_dir
        # Renormalize
        pts[nearest_sphere_idx] /= np.linalg.norm(pts[nearest_sphere_idx])
        
        # Check if we improved
        new_radius = _covering_radius_exact(pts, grid)
        if new_radius < current_radius:
            current_radius = new_radius
        else:
            # Revert if it got worse
            pts[nearest_sphere_idx] -= step_size * move_dir
            pts[nearest_sphere_idx] /= np.linalg.norm(pts[nearest_sphere_idx])
            # Try smaller step
            step_size *= 0.5
            if step_size < 1e-6:
                break
    
    return pts, current_radius


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    timeout = 115.0  # Leave 5s buffer
    
    # Initial points
    points = _initial_points()
    
    # Precompute grid for objective function - use larger grid for better accuracy
    grid = _spherical_grid(n_points=20000)
    
    best_points = points.copy()
    best_radius = _covering_radius_exact(points, grid)
    
    try:
        # Phase 1: Optimize with smooth objective using annealed temperature
        temperatures = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
        
        for temp in temperatures:
            if time.time() - start_time > timeout - 30:
                break
            
            def obj_smooth(x):
                return _smooth_objective(x, grid, temperature=temp)
            
            x0 = best_points.flatten()
            
            result = minimize(
                obj_smooth,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            if time.time() - start_time > timeout - 30:
                break
            
            opt_points = result.x.reshape(N, 3)
            norms = np.linalg.norm(opt_points, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-10)
            opt_points = opt_points / norms
            
            opt_radius = _covering_radius_exact(opt_points, grid)
            if opt_radius < best_radius:
                best_points = opt_points
                best_radius = opt_radius
        
        # Phase 2: Local refinement using subgradient method
        if time.time() - start_time < timeout - 10:
            refined_points, refined_radius = _local_refine(
                best_points, grid, max_iter=300, step_size=0.02
            )
            
            if refined_radius < best_radius:
                best_points = refined_points
                best_radius = refined_radius
        
        # Phase 3: Final fine-tuning with smooth objective
        if time.time() - start_time < timeout - 5:
            def obj_smooth_final(x):
                return _smooth_objective(x, grid, temperature=0.005)
            
            x0 = best_points.flatten()
            result2 = minimize(
                obj_smooth_final,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            
            if time.time() - start_time < timeout - 5:
                opt_points2 = result2.x.reshape(N, 3)
                norms = np.linalg.norm(opt_points2, axis=1, keepdims=True)
                norms = np.maximum(norms, 1e-10)
                opt_points2 = opt_points2 / norms
                
                opt_radius2 = _covering_radius_exact(opt_points2, grid)
                if opt_radius2 < best_radius:
                    best_points = opt_points2
                    best_radius = opt_radius2
                    
    except Exception:
        pass
    
    # Ensure points are on the unit sphere
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    best_points = best_points / norms
    
    # Check for coincident points and hemisphere issues
    tree = cKDTree(best_points)
    dists, _ = tree.query(best_points, k=2)
    if np.min(dists[:, 1]) < 1e-6:
        best_points = _initial_points()
    
    return best_points
# EVOLVE-BLOCK-END
