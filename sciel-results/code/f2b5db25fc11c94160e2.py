import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130
_EPS = 1e-12

def _compute_hull_radius(pts):
    """
    Compute the true covering radius using the convex hull.
    This is the ground truth for scoring, used only for final validation and best tracking.
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        # Compute normals for each face
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        
        # Filter degenerate
        valid = norms.flatten() > _EPS
        if not np.any(valid):
            return np.pi
            
        normals = cross[valid] / (norms[valid] + _EPS)
        
        # Ensure outward direction
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        
        # The pole is the normalized normal.
        # Distance is arccos(pole . vertex)
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        return np.max(radii)
    except Exception:
        return np.pi

def _grid_covering_radius(pts, n_theta=100, n_phi=200):
    """
    Fast approximation of covering radius using a spherical grid.
    This is much faster than ConvexHull and sufficient for guiding the optimization.
    """
    # Generate grid points
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing='ij')
    
    # Convert to Cartesian
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    
    grid_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    # Compute distances from all grid points to all N points
    # dists[i, j] = angular distance between grid_pts[i] and pts[j]
    # cos_dist = grid_pts @ pts.T
    cos_dist = grid_pts @ pts.T
    cos_dist = np.clip(cos_dist, -1.0, 1.0)
    
    # Min distance for each grid point
    min_cos = np.max(cos_dist, axis=1)
    min_dist = np.arccos(min_cos)
    
    return np.max(min_dist)

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_init(rng):
    """Generate random initial points."""
    pts = rng.standard_normal((N, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def construct():
    deadline = time.time() + 145.0
    rng = np.random.default_rng(42)
    
    # Start with Fibonacci spiral
    current_pts = _initial_fib()
    current_cost = _grid_covering_radius(current_pts)
    
    best_pts = current_pts.copy()
    best_cost = current_cost
    
    # Simulated Annealing Parameters
    # Initial temperature: large enough to accept many worse moves
    # Final temperature: small enough to converge
    T_start = 0.5
    T_end = 0.001
    n_iterations = 50000 # Will be limited by time
    
    # Pre-compute grid for fast evaluation? 
    # _grid_covering_radius creates grid every time. Let's optimize that.
    # For now, we'll rely on the vectorized operation being fast enough.
    # N=130, Grid=20000. 20000x130 matrix mult is ~2.6M ops. Very fast.
    
    t_iter = 0
    while time.time() < deadline - 10.0:
        t_iter += 1
        t = t_iter / n_iterations
        if t > 1.0:
            t = 1.0
        T = T_start * (T_end / T_start) ** t
        
        # Choose a random point to move
        idx = rng.integers(N)
        
        # Generate a random perturbation
        # Move in a random direction on the sphere
        # Step size can be annealed too, but fixed small step works well with SA
        step_size = 0.1 
        noise = rng.standard_normal(3)
        noise /= np.linalg.norm(noise)
        
        # New position: current + noise * step, then normalize
        new_pt = current_pts[idx] + noise * step_size
        new_pt /= np.linalg.norm(new_pt)
        
        # Create new configuration
        new_pts = current_pts.copy()
        new_pts[idx] = new_pt
        
        # Evaluate new cost
        new_cost = _grid_covering_radius(new_pts)
        
        # Acceptance probability
        delta_cost = new_cost - current_cost
        if delta_cost < 0 or rng.random() < np.exp(-delta_cost / T):
            current_pts = new_pts
            current_cost = new_cost
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_pts = current_pts.copy()
        
        # Occasionally, do a larger "jump" to escape local minima
        if t_iter % 1000 == 0:
             # Perturb a random subset of points
             n_perturb = 10
             indices = rng.choice(N, n_perturb, replace=False)
             for idx in indices:
                 noise = rng.standard_normal(3)
                 noise /= np.linalg.norm(noise)
                 new_pt = current_pts[idx] + noise * 0.5
                 new_pt /= np.linalg.norm(new_pt)
                 current_pts[idx] = new_pt
             current_cost = _grid_covering_radius(current_pts)
             if current_cost < best_cost:
                 best_cost = current_cost
                 best_pts = current_pts.copy()

    # Final polish: Use the best found so far and do a few iterations of local relaxation (Lloyd-like or gradient)
    # But first, let's verify the best point with the true hull radius
    final_pts = best_pts
    final_hull_r = _compute_hull_radius(final_pts)
    
    # If we have time, try to improve the best point with a local search (Gradient Descent on grid cost)
    # This helps refine the structure found by SA.
    if time.time() < deadline - 5.0:
        polish_deadline = time.time() + (deadline - 5.0 - time.time())
        pts = best_pts.copy()
        cost = _grid_covering_radius(pts)
        lr = 0.01
        
        # Simple gradient step on grid cost is hard to compute analytically fast.
        # Instead, do a few more SA iterations with very low T (essentially hill climbing)
        T_polish = 0.0001
        for _ in range(1000):
            if time.time() > polish_deadline:
                break
            idx = rng.integers(N)
            noise = rng.standard_normal(3)
            noise /= np.linalg.norm(noise)
            new_pt = pts[idx] + noise * 0.05
            new_pt /= np.linalg.norm(new_pt)
            
            new_pts = pts.copy()
            new_pts[idx] = new_pt
            new_cost = _grid_covering_radius(new_pts)
            
            if new_cost < cost:
                pts = new_pts
                cost = new_cost
        
        # Check if polished is better
        polished_hull_r = _compute_hull_radius(pts)
        if polished_hull_r < final_hull_r:
            final_pts = pts
            final_hull_r = polished_hull_r

    # Final normalization and safety check
    final_pts = np.array(final_pts, dtype=float)
    norms = np.linalg.norm(final_pts, axis=1, keepdims=True)
    final_pts /= (norms + _EPS)
    
    if not np.all(np.isfinite(final_pts)) or final_pts.shape != (N, 3):
        return _initial_fib()
        
    return final_pts
