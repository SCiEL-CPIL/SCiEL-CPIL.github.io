import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130
_EPS = 1e-12

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _get_grid():
    """
    Generate a high-resolution spherical grid for the surrogate objective.
    We use a Fibonacci lattice with ~20,000 points to ensure the grid spacing
    is significantly smaller than the expected covering radius (~11 degrees).
    """
    M = 20000
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_hull_radius(pts):
    """
    Compute the true covering radius using the convex hull.
    This is the ground truth for scoring.
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

def _soft_attention_objective_and_grad(pts_flat, grid, beta):
    """
    Objective: Minimize the maximum over grid points of the distance to the nearest point.
    Surrogate: Use a soft-min (log-sum-exp) to make it differentiable.
    f(pts) = log( sum_g exp(-beta * d(g, nearest(pts))^2) )
    Actually, to minimize max distance, we can minimize the sum of squared distances 
    weighted by a soft-max of the distances, or simply minimize the log-sum-exp of 
    squared distances.
    
    Let d_g = min_i ||g - p_i||^2.
    We want to minimize max_g d_g.
    A smooth approximation is (1/beta) * log( sum_g exp(beta * d_g) ).
    
    However, d_g itself is a min over i, which is non-differentiable.
    We can approximate d_g = min_i ||g - p_i||^2 using a soft-min:
    d_g_soft = - (1/beta) * log( sum_i exp(-beta * ||g - p_i||^2) )
    
    Then the objective is J = (1/beta) * log( sum_g exp(beta * d_g_soft) ).
    
    This is computationally expensive (M x N matrix).
    M=20000, N=130. 20000*130 = 2.6M elements. This is manageable in NumPy.
    """
    pts = pts_flat.reshape(N, 3)
    # Normalize points to sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / (norms + _EPS)
    
    # Compute squared distances between all grid points and all config points
    # grid: (M, 3), pts: (N, 3)
    # d_sq[g, i] = ||grid[g] - pts[i]||^2 = 2 - 2 * (grid[g] . pts[i])
    dots = grid @ pts.T  # (M, N)
    d_sq = 2.0 - 2.0 * dots
    
    # Soft-min over i for each g
    # d_g_soft = - (1/beta) * log( sum_i exp(-beta * d_sq[g, i]) )
    # To avoid overflow/underflow, subtract max
    max_exp = np.max(-beta * d_sq, axis=1, keepdims=True)
    exp_terms = np.exp(-beta * d_sq - max_exp)
    sum_exp = np.sum(exp_terms, axis=1)
    log_sum = np.log(sum_exp + _EPS) + max_exp.flatten()
    d_g_soft = -log_sum / beta
    
    # Objective: Smooth max over g
    # J = (1/beta) * log( sum_g exp(beta * d_g_soft) )
    max_d = np.max(d_g_soft)
    exp_d = np.exp(beta * (d_g_soft - max_d))
    sum_d = np.sum(exp_d)
    J = (max_d + np.log(sum_d + _EPS)) / beta
    
    # Gradient
    # dJ/d(pts) = ...
    # Let's use automatic differentiation logic or manual chain rule.
    # Given complexity, we might rely on scipy's numerical gradient if analytical is too slow to derive/code correctly here.
    # But numerical gradient for 390 variables is 390 evaluations of J.
    # J involves 20000x130 ops. 390 * 2.6M ops ~ 1B ops. Might be too slow for 150s budget if we do many iterations.
    
    # Alternative: Use the gradient of the soft-min directly on the "worst" grid points?
    # Or just use a simpler objective: Sum of squared distances to nearest neighbor, weighted by distance?
    # Let's try a simpler surrogate: Minimize sum_g (d_g_soft)^2.
    # This pushes all grid points closer.
    
    # Let's stick to the log-sum-exp max approximation but compute gradient numerically with a smaller grid?
    # No, let's use a simpler gradient-based method:
    # Move each point p_i in the direction that reduces the distances of the grid points "assigned" to it.
    # Assignment: g is assigned to i if i is the argmin of d_sq[g, i].
    # This is Lloyd's algorithm, which is not differentiable but very effective.
    
    # So, let's implement Lloyd's Algorithm (Voronoi Relaxation) as the main optimizer.
    # It is deterministic and often faster than gradient descent for covering problems.
    return J, None # Placeholder, we will use Lloyd's

def _lloyd_relax(pts, grid, n_iters=50):
    """
    Lloyd's Algorithm for spherical covering.
    1. Compute Voronoi cells (approximated by nearest neighbor on grid).
    2. Move each point to the centroid of its cell.
    """
    pts = pts.copy()
    for _ in range(n_iters):
        # Compute nearest point for each grid point
        dots = grid @ pts.T  # (M, N)
        # Nearest point corresponds to max dot product
        nearest_idx = np.argmax(dots, axis=1)
        
        # Compute centroids
        new_pts = np.zeros_like(pts)
        counts = np.zeros(N)
        
        # Use bincount for efficiency
        for i in range(N):
            mask = nearest_idx == i
            if np.any(mask):
                new_pts[i] = np.sum(grid[mask], axis=0)
                counts[i] = np.sum(mask)
        
        # Normalize centroids
        valid = counts > 0
        new_pts[valid] /= (np.linalg.norm(new_pts[valid], axis=1, keepdims=True) + _EPS)
        
        # If a point has no assigned grid points, keep it or randomize?
        # In a dense grid, this shouldn't happen unless points are very close.
        # If it happens, keep original.
        new_pts[~valid] = pts[~valid]
        
        # Check for convergence
        diff = np.linalg.norm(new_pts - pts)
        pts = new_pts
        if diff < 1e-6:
            break
            
    return pts

def _local_polish(pts, deadline):
    """
    Perform a local polish using gradient-free optimization (Nelder-Mead) on small perturbations
    or just more Lloyd iterations with a finer grid?
    Lloyd's is usually enough. Let's do a final check with a very fine grid if time permits.
    """
    # Use a finer grid for the final Lloyd iteration
    M_fine = 50000
    i = np.arange(M_fine) + 0.5
    z = 1.0 - 2.0 * i / M_fine
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    grid_fine = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Run Lloyd's on fine grid
    pts = _lloyd_relax(pts, grid_fine, n_iters=20)
    
    return pts

def construct():
    deadline = time.time() + 145.0
    
    # Start with Fibonacci
    best_pts = _initial_fib()
    best_r = _compute_hull_radius(best_pts)
    
    # Multi-restart with Random Initializations
    rng = np.random.default_rng(42)
    n_restarts = 10
    
    # Allocate time
    time_per_restart = (deadline - 10.0) / (n_restarts + 1)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
            
        if restart == 0:
            pts = _initial_fib()
        else:
            pts = rng.standard_normal((N, 3))
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            
        # Run Lloyd's Algorithm
        # Use a medium grid for speed during restarts
        M_med = 10000
        i = np.arange(M_med) + 0.5
        z = 1.0 - 2.0 * i / M_med
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
        grid_med = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        # Limit iterations based on time
        start_t = time.time()
        max_iters = 100
        for _ in range(max_iters):
            if time.time() > start_t + time_per_restart * 0.8:
                break
            pts = _lloyd_relax(pts, grid_med, n_iters=5) # 5 iters per step
            # Check improvement
            curr_r = _compute_hull_radius(pts)
            if curr_r < best_r:
                best_r = curr_r
                best_pts = pts.copy()
                
    # Final polish on the best configuration
    if time.time() < deadline - 5.0:
        best_pts = _local_polish(best_pts, deadline)
        final_r = _compute_hull_radius(best_pts)
        if final_r < best_r:
            best_r = final_r
            # best_pts is already updated
            
    # Final normalization
    best_pts = np.array(best_pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
