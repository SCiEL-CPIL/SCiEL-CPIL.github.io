import time
import numpy as np
from scipy.spatial import ConvexHull
import random

N = 130
_EPS = 1e-12

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _get_grid(n_grid=20000):
    """Generate a dense spherical grid for the proxy objective."""
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _hull_radius(pts):
    """Compute the exact covering radius using the convex hull method."""
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        area = 0.5 * np.linalg.norm(cross, axis=1)
        
        valid = area > _EPS
        if not np.any(valid):
            return np.pi

        normals = cross[valid]
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1, keepdims=True)
        normals = normals * np.sign(signs)
        normals = normals / (norms + _EPS)
        
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        
        return np.max(radii)
    except Exception:
        return np.pi

def _proxy_radius(pts, grid):
    """
    Fast proxy for covering radius: max over grid points of min distance to any point.
    Returns angle in radians.
    """
    # dist_sq[i, j] = ||grid[i] - pts[j]||^2 = 2 - 2 dot(grid[i], pts[j])
    dots = grid @ pts.T
    dist_sq = 2.0 - 2.0 * dots
    min_dist_sq = np.min(dist_sq, axis=1)
    max_dist_sq = np.max(min_dist_sq)
    # Convert squared chord distance to angle: d^2 = 2(1-cos(theta)) => cos(theta) = 1 - d^2/2
    cos_theta = 1.0 - max_dist_sq / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)

def _rgd_step(pts, grid, step_size, steps=10):
    """
    Perform a few steps of Riemannian Gradient Descent to polish.
    Uses the proxy objective gradient.
    """
    for _ in range(steps):
        # Compute proxy objective and gradient
        dots = grid @ pts.T
        dist_sq = 2.0 - 2.0 * dots
        min_dist_sq = np.min(dist_sq, axis=1)
        
        # Gradient of min_dist_sq w.r.t pts
        # f_i = min_j (2 - 2 g_i . p_j)
        # d f_i / d p_k = -2 g_i if k is argmin, else 0
        min_idx = np.argmin(dist_sq, axis=1)
        
        # We want to minimize max_i f_i.
        # Approximate gradient: push points away from grid points that are "far" from them.
        # Specifically, for each point k, sum the contributions from grid points i where k is the nearest.
        # weight_i = 1 if i is in the "active set" (near max), else 0.
        # For simplicity, use uniform weights over all grid points, but scaled by how close f_i is to max.
        # This is a subgradient-like approximation.
        
        # Let's use a simpler heuristic gradient:
        # Move each point towards the centroid of the grid points for which it is the nearest,
        # but weighted by the distance error.
        
        grad = np.zeros_like(pts)
        # Accumulate -2 * grid[i] for each i into grad[min_idx[i]]
        # This is the gradient of sum(f_i). We want to minimize max(f_i).
        # A common trick: weight by softmax of f_i / T.
        
        T = 0.01 # Temperature for softmax weighting
        scaled = min_dist_sq / T
        max_scaled = np.max(scaled)
        exp_vals = np.exp(scaled - max_scaled)
        weights = exp_vals / np.sum(exp_vals)
        
        weighted_grid = -2.0 * grid * weights[:, np.newaxis]
        np.add.at(grad, min_idx, weighted_grid)
        
        # Project to tangent space
        dot_prod = np.sum(grad * pts, axis=1, keepdims=True)
        grad_tan = grad - dot_prod * pts
        
        # Normalize and step
        grad_norm = np.linalg.norm(grad_tan, axis=1, keepdims=True)
        grad_tan = grad_tan / (grad_norm + _EPS)
        
        new_pts = pts - step_size * grad_tan
        new_pts /= (np.linalg.norm(new_pts, axis=1, keepdims=True) + _EPS)
        
        # Check if it improved the proxy
        old_r = _proxy_radius(pts, grid)
        new_r = _proxy_radius(new_pts, grid)
        
        if new_r < old_r:
            pts = new_pts
        else:
            # If RGD step didn't help, break early
            break
            
    return pts

def _simulated_annealing_search(pts, grid, deadline, rng):
    """
    Combinatorial Annealing:
    1. Pick a random point index i.
    2. Generate a candidate point p_new (perturbed random or from a local pool).
    3. Evaluate proxy radius for pts with i replaced by p_new.
    4. Accept if better, or with probability exp(-dE/T).
    """
    best_pts = pts.copy()
    best_proxy = _proxy_radius(pts, grid)
    current_proxy = best_proxy
    
    T_start = 0.1
    T_end = 0.001
    total_iters = 5000 # Target number of iterations, limited by time
    
    # Pre-compute a set of random candidate points to choose from?
    # Or just generate on the fly? On the fly is fine.
    
    start_time = time.time()
    i_iter = 0
    
    while time.time() < deadline and i_iter < total_iters:
        i_iter += 1
        
        # Adaptive temperature
        progress = i_iter / total_iters
        T = T_start * (T_end / T_start) ** progress
        
        # Pick random index to move
        idx = rng.integers(0, N)
        
        # Generate candidate:
        # 10% chance: fully random point
        # 90% chance: perturbation of current point
        if rng.random() < 0.1:
            cand = rng.standard_normal(3)
            cand /= np.linalg.norm(cand)
        else:
            # Small perturbation
            noise = rng.standard_normal(3) * 0.1
            cand = pts[idx] + noise
            cand /= np.linalg.norm(cand)
        
        # Evaluate
        old_pts = pts[idx].copy()
        pts[idx] = cand
        
        new_proxy = _proxy_radius(pts, grid)
        dE = new_proxy - current_proxy
        
        if dE < 0 or rng.random() < np.exp(-dE / T):
            # Accept
            current_proxy = new_proxy
            if current_proxy < best_proxy:
                best_proxy = current_proxy
                best_pts = pts.copy()
        else:
            # Reject
            pts[idx] = old_pts
            
        # Periodic polish with RGD
        if i_iter % 200 == 0:
            # Time check before polishing
            if time.time() > deadline - 2.0:
                break
            pts = _rgd_step(pts, grid, step_size=0.05, steps=5)
            new_proxy = _proxy_radius(pts, grid)
            if new_proxy < current_proxy:
                current_proxy = new_proxy
                if current_proxy < best_proxy:
                    best_proxy = current_proxy
                    best_pts = pts.copy()
        
    return best_pts

def construct():
    deadline = time.time() + 145.0  # Safety margin (150s limit)
    grid = _get_grid(20000)
    
    best_pts = _initial_fib()
    best_proxy = _proxy_radius(best_pts, grid)
    
    # Phase 1: Multiple restarts of Simulated Annealing
    # We use 3 restarts. 
    # Restart 0: Fib
    # Restart 1: Fib + noise
    # Restart 2: Best from Restart 0 + noise
    
    n_restarts = 3
    time_per_restart = (deadline - 30.0) / n_restarts # Leave 30s for final polish
    
    rng = np.random.default_rng(42)
    
    for restart in range(n_restarts):
        if time.time() > deadline - 30.0:
            break
        
        if restart == 0:
            pts = _initial_fib()
        elif restart == 1:
            pts = _initial_fib() + rng.standard_normal((N, 3)) * 0.1
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        else:
            # Perturb the best found so far
            pts = best_pts + rng.standard_normal((N, 3)) * 0.1
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            
        # Run SA
        # Allocate remaining time for this restart, minus buffer for next restarts and polish
        # Simplified: give it time_per_restart
        end_time = time.time() + time_per_restart
        new_pts = _simulated_annealing_search(pts, grid, end_time, rng)
        
        new_proxy = _proxy_radius(new_pts, grid)
        if new_proxy < best_proxy:
            best_proxy = new_proxy
            best_pts = new_pts
    
    # Phase 2: Final Polish
    # Use RGD on the best candidate found
    if time.time() < deadline - 5.0:
        # Intensive RGD polish
        # We can run RGD for the remaining time
        while time.time() < deadline - 2.0:
            # Try a few RGD steps
            polished = _rgd_step(best_pts, grid, step_size=0.02, steps=20)
            new_proxy = _proxy_radius(polished, grid)
            if new_proxy < best_proxy:
                best_proxy = new_proxy
                best_pts = polished
            else:
                # If no improvement, try a small random perturbation to escape
                if rng.random() < 0.1:
                    idx = rng.integers(0, N)
                    noise = rng.standard_normal(3) * 0.05
                    best_pts[idx] += noise
                    best_pts[idx] /= np.linalg.norm(best_pts[idx])
                else:
                    break # Stagnation, break
    
    # Final normalization and validity check
    best_pts = np.array(best_pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
