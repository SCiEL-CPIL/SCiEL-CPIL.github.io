"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = _START_TIME + 140.0  # Safety margin before 150s kill

def _to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _get_hull_data(pts):
    """
    Compute convex hull, facet circumradii, and circumcenters.
    Returns:
        hull: ConvexHull object
        radii: array of circumradii (angular) for each facet
        centers: array of circumcenter unit vectors for each facet
        valid_mask: boolean mask for valid facets
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None, None, None
    
    n_facets = len(hull.simplices)
    radii = np.zeros(n_facets)
    centers = np.zeros((n_facets, 3))
    valid_mask = np.ones(n_facets, dtype=bool)
    
    for i, simplex in enumerate(hull.simplices):
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        v1 = b - a
        v2 = c - a
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        
        if norm_n < 1e-10:
            valid_mask[i] = False
            continue
            
        n = n / norm_n
        
        # Determine sign of normal: should point away from origin (outward)
        # For a convex hull containing the origin, the outward normal n satisfies n . a > 0
        # Actually, for a sphere, the facet is part of the boundary.
        # The circumcenter is the point on the sphere equidistant to a, b, c.
        # It lies along the normal line.
        # We need the one that is the "center" of the spherical triangle.
        # Check dot product with centroid.
        centroid = (a + b + c) / 3.0
        if np.dot(n, centroid) < 0:
            n = -n
            
        centers[i] = n
        dot_val = np.clip(np.dot(n, a), -1.0, 1.0)
        radii[i] = np.arccos(dot_val)
        
    return hull, radii, centers, valid_mask

def _objective_and_grad(x_flat):
    """
    Objective: Smooth approximation of max circumradius using log-sum-exp.
    Gradient: Numerical or analytical approximation.
    For stability, we use a numerical gradient with a small step, 
    but we compute the objective analytically.
    """
    pts = _to_sphere(x_flat.reshape(N, 3))
    
    hull, radii, centers, valid_mask = _get_hull_data(pts)
    if hull is None:
        return 100.0, np.zeros_like(x_flat)
        
    if not np.any(valid_mask):
        return 100.0, np.zeros_like(x_flat)
        
    valid_radii = radii[valid_mask]
    
    # Log-sum-exp for smooth max
    # L = alpha * log(sum(exp(radii/alpha)))
    # As alpha -> 0, this approaches max(radii)
    # We use a fixed alpha for smoothness during optimization
    alpha = 0.05 
    max_r = np.max(valid_radii)
    # Shift for numerical stability
    shifted = valid_radii - max_r
    log_sum_exp = max_r + alpha * np.log(np.sum(np.exp(shifted / alpha)))
    
    # Numerical Gradient
    # L-BFGS-B will use its own numerical gradient if we don't provide one,
    # but providing one can help. However, the topology of the hull changes
    # discretely, making the gradient noisy.
    # It is often safer to let L-BFGS-B compute the gradient numerically 
    # over the smooth objective, but the objective itself is non-smooth 
    # at hull topology changes.
    # We will return the objective and let scipy handle the gradient,
    # but we must ensure the function is consistent.
    
    return log_sum_exp, None

def _run_lbfgs(pts_init, time_limit=20.0):
    """
    Run L-BFGS-B optimization starting from pts_init.
    """
    x0 = pts_init.flatten()
    
    # We use 'L-BFGS-B' with finite difference gradient.
    # The objective is the smooth log-sum-exp of facet radii.
    
    start_t = time.time()
    
    def fun_and_grad(x):
        val, _ = _objective_and_grad(x)
        return val, None # Let scipy approximate gradient
        
    # Options for L-BFGS-B
    options = {
        'maxiter': 200,
        'ftol': 1e-8,
        'gtol': 1e-5,
        'maxls': 20,
        'maxcor': 10,
        'disp': False
    }
    
    try:
        res = minimize(fun_and_grad, x0, method='L-BFGS-B', options=options)
    except Exception:
        return pts_init
        
    if time.time() - start_t > time_limit:
        return pts_init
        
    if res.success and res.fun < _objective_and_grad(x0)[0]:
        return _to_sphere(res.x.reshape(N, 3))
    return pts_init

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    # 1. Generate multiple diverse initial configurations
    # Strategy: Use a mix of Fibonacci spiral, random, and slightly perturbed known good configs
    # Since we don't have the HSS 1994 config hardcoded, we rely on good random starts + optimization.
    
    initializations = []
    
    # Init 1: Fibonacci Spiral (Standard good start)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    initializations.append(_to_sphere(fib_pts))
    
    # Init 2: Random Uniform
    np.random.seed(42)
    rand_pts = np.random.randn(N, 3)
    initializations.append(_to_sphere(rand_pts))
    
    # Init 3: Perturbed Fibonacci
    fib_pert = _to_sphere(fib_pts + np.random.randn(N, 3) * 0.05)
    initializations.append(fib_pert)
    
    # Init 4: Another random with different seed
    np.random.seed(123)
    rand_pts2 = np.random.randn(N, 3)
    initializations.append(_to_sphere(rand_pts2))
    
    best_pts = None
    best_score = np.inf
    
    deadline = _START_TIME + 140.0
    
    # Evaluate and optimize each initialization
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 15:
            break
            
        # Quick local polish before global LBFGS to get into a good basin
        # A few steps of simple Lloyd's or just random restarts in LBFGS
        current_pts = init_pts.copy()
        
        # Run L-BFGS
        time_left_for_opt = min(25.0, deadline - time.time() - 5.0)
        if time_left_for_opt > 5:
            opt_pts = _run_lbfgs(current_pts, time_limit=time_left_for_opt)
        else:
            opt_pts = current_pts
            
        # Evaluate true covering radius (max facet radius)
        hull, radii, centers, valid_mask = _get_hull_data(opt_pts)
        if hull is not None and np.any(valid_mask):
            true_score = np.max(radii[valid_mask])
        else:
            true_score = np.pi
            
        if true_score < best_score:
            best_score = true_score
            best_pts = opt_pts.copy()
            
        # If we have time, try a second pass with different alpha or restart
        if time.time() < deadline - 30:
             # Restart from best found so far with a small perturbation
             if best_pts is not None:
                 restart_pts = _to_sphere(best_pts + np.random.randn(N, 3) * 0.01)
                 time_left_for_opt2 = min(15.0, deadline - time.time() - 5.0)
                 if time_left_for_opt2 > 5:
                     opt_pts2 = _run_lbfgs(restart_pts, time_limit=time_left_for_opt2)
                     hull2, radii2, centers2, valid_mask2 = _get_hull_data(opt_pts2)
                     if hull2 is not None and np.any(valid_mask2):
                         score2 = np.max(radii2[valid_mask2])
                         if score2 < best_score:
                             best_score = score2
                             best_pts = opt_pts2.copy()
                             
    # Final validation
    if best_pts is None:
        return _to_sphere(fib_pts)
        
    best_pts = _to_sphere(best_pts)
    
    # Check for coincident points
    dots = best_pts @ best_pts.T
    np.fill_diagonal(dots, 0)
    if np.max(np.abs(dots)) > 0.999999:
        best_pts = _to_sphere(best_pts + np.random.randn(N, 3) * 0.01)
        
    return best_pts
# EVOLVE-BLOCK-END
