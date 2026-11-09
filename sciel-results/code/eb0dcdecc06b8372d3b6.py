"""
Covering the sphere with 50 equal caps.
Uses convex hull facet circumradii for exact evaluation.
Structural change: use iterative facet-based local refinement with adaptive steps,
combined with a lightweight smooth surrogate for global optimization.
"""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _exact_covering_radius(pts):
    """Compute the exact covering radius in radians using convex hull facet circumradii."""
    try:
        hull = ConvexHull(pts)
        max_r = 0.0
        for simplex in hull.simplices:
            v0, v1, v2 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            ab = v1 - v0
            ac = v2 - v0
            n = np.cross(ab, ac)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            d = np.abs(np.dot(n, v0))
            r = np.arccos(np.clip(d, -1.0, 1.0))
            if r > max_r:
                max_r = r
        return max_r
    except Exception:
        return np.pi

def _lightweight_energy(pts, beta=20.0, M=1000):
    """
    Lightweight smooth surrogate for fast gradient-based optimization.
    Uses a smaller grid and simplified soft-max.
    """
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    dots = grid @ pts.T
    dots = np.clip(dots, -1.0 + 1e-12, 1.0 - 1e-12)
    
    # For each grid point, find the nearest point's dot product
    # Use a simplified approach: for each grid point, take the max dot
    max_dots = np.max(dots, axis=1)
    max_dots = np.clip(max_dots, -1.0 + 1e-12, 1.0 - 1e-12)
    d = np.arccos(max_dots)
    
    # Soft-max over grid points to get a smooth approximation of the max
    shifted = beta * d
    max_shifted = np.max(shifted)
    exp_shifted = np.exp(shifted - max_shifted)
    weights = exp_shifted / np.sum(exp_shifted)
    E = np.sum(weights * d)
    
    # Gradient: dE/dpts_j = sum_y weights[y] * (d/dpts_j) max_i(grid_y . pts_i)
    # The gradient of max_i(grid_y . pts_i) w.r.t. pts_j is grid_y if j is the argmax, else 0
    # For a smooth approximation, use the weights from the soft-max over i
    shifted_i = beta * dots  # (M, N)
    max_shifted_i = np.max(shifted_i, axis=1, keepdims=True)
    exp_i = np.exp(shifted_i - max_shifted_i)
    weights_i = exp_i / np.sum(exp_i, axis=1, keepdims=True)  # (M, N)
    
    # dE/dpts_j = sum_y weights[y] * weights_i[y, j] * grid[y] * (-1/sqrt(1 - max_dots[y]^2))
    # Actually, d(arccos(max_dot))/d(pts_j) = -1/sqrt(1-max_dot^2) * d(max_dot)/d(pts_j)
    # d(max_dot)/d(pts_j) ≈ weights_i[y, j] * grid[y]
    
    dg_ddots = -weights / np.sqrt(np.maximum(1.0 - max_dots * max_dots, 1e-12))  # (M,)
    weighted = weights_i * dg_ddots[:, None]  # (M, N)
    grad = weighted.T @ grid  # (N, 3)
    
    return E, grad

def _local_refine(pts, max_iters=50, step_init=0.05, step_decay=0.95):
    """
    Iterative local refinement: repeatedly find worst facet and move its vertices.
    Returns improved points and the covering radius.
    """
    current_pts = pts.copy()
    current_radius = _exact_covering_radius(current_pts)
    step = step_init
    stagnant = 0
    
    for it in range(max_iters):
        try:
            hull = ConvexHull(current_pts)
            max_r = 0.0
            worst_simplex = None
            for simplex in hull.simplices:
                v0, v1, v2 = current_pts[simplex[0]], current_pts[simplex[1]], current_pts[simplex[2]]
                ab = v1 - v0
                ac = v2 - v0
                n = np.cross(ab, ac)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-12:
                    continue
                n = n / n_norm
                d = np.abs(np.dot(n, v0))
                r = np.arccos(np.clip(d, -1.0, 1.0))
                if r > max_r:
                    max_r = r
                    worst_simplex = simplex
            
            if worst_simplex is None:
                break
            
            # Move the three vertices of the worst facet towards the circumcenter
            v0, v1, v2 = current_pts[worst_simplex[0]], current_pts[worst_simplex[1]], current_pts[worst_simplex[2]]
            ab = v1 - v0
            ac = v2 - v0
            n = np.cross(ab, ac)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                break
            n = n / n_norm
            
            # Adaptive step: smaller when radius is small
            adaptive_step = step * (current_radius / (np.pi / 6))
            adaptive_step = np.clip(adaptive_step, 0.001, 0.1)
            
            for idx in worst_simplex:
                current_pts[idx] += adaptive_step * n
                current_pts[idx] /= np.linalg.norm(current_pts[idx])
            
            new_radius = _exact_covering_radius(current_pts)
            if new_radius < current_radius:
                current_radius = new_radius
                stagnant = 0
            else:
                stagnant += 1
                step *= step_decay
                if stagnant > 10:
                    break
        except Exception:
            break
    
    return current_pts, current_radius

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Initial configuration
    best_pts = _fibonacci_sphere(N)
    best_radius = _exact_covering_radius(best_pts)
    
    # Try multiple initializations with different strategies
    rng = np.random.default_rng(42)
    inits = [best_pts.copy()]
    
    # Add some structured initializations
    for offset in [0.1, 0.2, 0.3]:
        perturbed = best_pts.copy()
        noise = rng.standard_normal((N, 3)) * offset
        perturbed += noise
        perturbed /= np.linalg.norm(perturbed, axis=1, keepdims=True)
        inits.append(perturbed)
    
    # Random initializations
    for _ in range(3):
        random_pts = rng.standard_normal((N, 3))
        random_pts /= np.linalg.norm(random_pts, axis=1, keepdims=True)
        inits.append(random_pts)
    
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 25:
            break
        
        current_pts = init_pts.copy()
        
        # Phase 1: Lightweight gradient-based optimization
        def obj_and_grad(x):
            pts = x.reshape(N, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            pts = pts / norms
            E, grad = _lightweight_energy(pts)
            # Project gradient to tangent space
            grad = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
            return E, grad.ravel()
        
        x0 = current_pts.ravel()
        try:
            result = minimize(obj_and_grad, x0, method='L-BFGS-B', 
                              options={'maxiter': 100, 'gtol': 1e-5})
            if result.fun < best_radius or result.success:
                current_pts = result.x.reshape(N, 3)
                norms = np.linalg.norm(current_pts, axis=1, keepdims=True)
                current_pts = current_pts / np.maximum(norms, 1e-12)
        except Exception:
            pass
        
        # Phase 2: Local refinement with facet-based moves
        if time.time() > deadline - 10:
            break
        
        refined_pts, refined_radius = _local_refine(current_pts, max_iters=80, step_init=0.04)
        
        if refined_radius < best_radius:
            best_radius = refined_radius
            best_pts = refined_pts.copy()
        
        # Phase 3: Additional polish with smaller steps
        if time.time() > deadline - 5:
            break
        
        polished_pts, polished_radius = _local_refine(best_pts, max_iters=50, step_init=0.015)
        if polished_radius < best_radius:
            best_radius = polished_radius
            best_pts = polished_pts.copy()
    
    # Final renormalization
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / np.maximum(norms, 1e-12)
    
    return best_pts
