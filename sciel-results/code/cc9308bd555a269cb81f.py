import numpy as np
from scipy.optimize import minimize
import time

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return pts / norms

def _fibonacci_spiral(n, seed=42):
    """Generate n points on unit sphere using Fibonacci spiral with optional jitter."""
    np.random.seed(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    # Add small random perturbation
    for idx in range(n):
        p = pts[idx]
        rand_vec = np.random.randn(3)
        tangent = rand_vec - np.dot(rand_vec, p) * p
        tangent_norm = np.linalg.norm(tangent)
        if tangent_norm > 1e-10:
            pts[idx] = p + 0.01 * tangent / tangent_norm
            pts[idx] = pts[idx] / np.linalg.norm(pts[idx])
    return _normalize(pts)

def _random_sphere(n, seed=0):
    """Generate n random points on the unit sphere."""
    np.random.seed(seed)
    pts = np.random.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return pts / norms

def _covering_radius_from_grid(pts, grid_pts):
    """
    Compute covering radius using a fixed grid of points on the sphere.
    For each grid point, find the nearest configuration point and compute angular distance.
    Returns the maximum angular distance in radians.
    """
    # Compute dot products between all grid points and all configuration points
    # Shape: (n_grid, n_config)
    dots = grid_pts @ pts.T
    # For each grid point, find the max dot product (nearest point)
    max_dots = np.max(dots, axis=1)
    # Angular distance = arccos(max_dot)
    distances = np.arccos(np.clip(max_dots, -1.0, 1.0))
    return np.max(distances)

def _covering_radius_gradient(pts, grid_pts):
    """
    Compute the covering radius and its gradient with respect to pts.
    The covering radius is the max over grid points of arccos(max_i dot(g, p_i)).
    
    For the grid point g* that achieves the max, and the nearest config point p_j*,
    the gradient of arccos(dot(g*, p_j*)) w.r.t. p_j* is:
      -g* / sqrt(1 - dot(g*, p_j*)^2)
    
    This is a subgradient approach - we use the active constraint.
    """
    dots = grid_pts @ pts.T  # (n_grid, n_config)
    max_dots = np.max(dots, axis=1)  # (n_grid,)
    max_idx = np.argmax(dots, axis=1)  # (n_grid,) index of nearest config point for each grid point
    
    distances = np.arccos(np.clip(max_dots, -1.0, 1.0))
    max_dist = np.max(distances)
    active_grid_idx = np.argmax(distances)
    
    # Gradient: only the active grid point and its nearest config point contribute
    g_active = grid_pts[active_grid_idx]
    j_active = max_idx[active_grid_idx]
    
    dot_val = np.dot(g_active, pts[j_active])
    dot_val = np.clip(dot_val, -1.0 + 1e-10, 1.0 - 1e-10)
    
    # Gradient of arccos(x) w.r.t. x is -1/sqrt(1-x^2)
    # d/dp arccos(g . p) = -g / sqrt(1 - (g.p)^2)
    denom = np.sqrt(1.0 - dot_val * dot_val)
    grad = np.zeros_like(pts)
    grad[j_active] = -g_active / denom
    
    return max_dist, grad

def _opti_cov(pts, deadline, verbose=False):
    """
    Optimize points to minimize the covering radius using gradient descent
    on the grid-based covering radius function.
    """
    # Generate a high-resolution grid for the sphere
    # Use an icosahedral subdivision or Fibonacci grid with many points
    n_grid = 5000
    grid_pts = _fibonacci_spiral(n_grid, seed=12345)
    
    current_pts = pts.copy()
    best_pts = pts.copy()
    best_radius = _covering_radius_from_grid(current_pts, grid_pts)
    
    step_size = 0.02
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline:
        # Compute radius and gradient
        radius, grad = _covering_radius_gradient(current_pts, grid_pts)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        grad = grad / grad_norm
        
        # Move in negative gradient direction, projected to tangent space
        new_pts = current_pts.copy()
        for idx in range(N):
            p = current_pts[idx]
            # Project gradient to tangent plane
            g_tangent = grad[idx] - np.dot(grad[idx], p) * p
            # Update
            new_pts[idx] = p - step_size * g_tangent
            # Renormalize
            norm = np.linalg.norm(new_pts[idx])
            if norm < 1e-10:
                new_pts[idx] = p
            else:
                new_pts[idx] = new_pts[idx] / norm
        
        # Evaluate new configuration
        new_radius = _covering_radius_from_grid(new_pts, grid_pts)
        
        if new_radius < best_radius - 1e-9:
            best_radius = new_radius
            best_pts = new_pts.copy()
            current_pts = new_pts.copy()
            no_improve_count = 0
            step_size = min(step_size * 1.02, 0.05)
        else:
            no_improve_count += 1
            step_size = max(step_size * 0.95, 0.005)
            
            if no_improve_count >= max_no_improve:
                # Try random restart with perturbation
                if time.time() < deadline - 10.0:
                    current_pts = best_pts.copy()
                    n_perturb = max(5, N // 10)
                    idxs = np.random.choice(N, n_perturb, replace=False)
                    for idx in idxs:
                        p = current_pts[idx]
                        rand_vec = np.random.randn(3)
                        tangent = rand_vec - np.dot(rand_vec, p) * p
                        tangent_norm = np.linalg.norm(tangent)
                        if tangent_norm > 1e-10:
                            current_pts[idx] = p + 0.03 * tangent / tangent_norm
                            current_pts[idx] = current_pts[idx] / np.linalg.norm(current_pts[idx])
                    no_improve_count = 0
                    step_size = 0.02
                else:
                    break
    
    return best_pts

def _lbfgs_polish(pts, deadline):
    """
    Use L-BFGS-B to polish the configuration by minimizing the grid-based covering radius.
    """
    n_grid = 3000
    grid_pts = _fibonacci_spiral(n_grid, seed=99999)
    
    def objective_and_grad(x):
        pts_local = _normalize(x.reshape(N, 3))
        radius, grad = _covering_radius_gradient(pts_local, grid_pts)
        # Convert gradient back to flat form
        grad_flat = np.zeros_like(x)
        for i in range(N):
            p = pts_local[i]
            g = grad[i]
            # Project to tangent space for the constraint
            g_tangent = g - np.dot(g, p) * p
            grad_flat[3*i:3*i+3] = g_tangent
        return radius, grad_flat
    
    x0 = pts.reshape(-1).copy()
    
    result = minimize(
        objective_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    if result.success or result.fun < _covering_radius_from_grid(pts, grid_pts):
        new_pts = _normalize(result.x.reshape(N, 3))
        return new_pts
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Try multiple initializations and pick the best
    best_overall = None
    best_overall_radius = np.inf
    
    # Initialization 1: Fibonacci spiral
    pts1 = _fibonacci_spiral(N, seed=42)
    r1 = _covering_radius_from_grid(pts1, _fibonacci_spiral(5000, seed=12345))
    if r1 < best_overall_radius:
        best_overall_radius = r1
        best_overall = pts1.copy()
    
    # Initialization 2: Random
    pts2 = _random_sphere(N, seed=7)
    r2 = _covering_radius_from_grid(pts2, _fibonacci_spiral(5000, seed=12345))
    if r2 < best_overall_radius:
        best_overall_radius = r2
        best_overall = pts2.copy()
    
    # Initialization 3: Fibonacci with different seed
    pts3 = _fibonacci_spiral(N, seed=123)
    r3 = _covering_radius_from_grid(pts3, _fibonacci_spiral(5000, seed=12345))
    if r3 < best_overall_radius:
        best_overall_radius = r3
        best_overall = pts3.copy()
    
    # Optimize the best initialization
    try:
        optimized = _opti_cov(best_overall, deadline)
        if np.all(np.isfinite(optimized)):
            # Polish with L-BFGS if time remains
            if time.time() < deadline - 20.0:
                polished = _lbfgs_polish(optimized, deadline)
                if np.all(np.isfinite(polished)):
                    optimized = polished
            
            # Final check
            final_radius = _covering_radius_from_grid(optimized, _fibonacci_spiral(5000, seed=12345))
            if final_radius < best_overall_radius:
                best_overall = optimized
                best_overall_radius = final_radius
    except Exception:
        pass
    
    return _normalize(best_overall)
