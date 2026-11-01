import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _energy_fsum(pts):
    """Compute logarithmic energy using math.fsum for precision."""
    pts = np.asarray(pts)
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)

def _energy_grad_vec(flat):
    """
    Compute energy and gradient for flattened points on sphere.
    Returns (energy, grad) where grad is (3*N,) tangent-projected gradient.
    """
    pts = flat.reshape(N, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    pts = pts / norms
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    dist_sq = np.maximum(dist_sq, 1e-15)
    
    mask = ~np.eye(N, dtype=bool)
    
    # Energy = sum_{i<j} log(1/d_ij) = -0.5 * sum_{i!=j} log(d_ij) = -0.25 * sum_{i!=j} log(d_ij^2)
    dsq_masked = np.where(mask, dist_sq, 1.0)
    energy = -0.25 * np.sum(np.log(dsq_masked))
    
    # Gradient: dE/dx_i = sum_{j!=i} (x_i - x_j) / d_ij^2
    inv_dsq = np.where(mask, 1.0 / dist_sq, 0.0)
    grad = np.sum(diff * inv_dsq[:, :, np.newaxis], axis=1)
    
    # Project to tangent space
    grad = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
    
    return energy, grad.ravel()

def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _riemannian_cg(pts, max_iter=200, tol=1e-10, time_limit=20.0):
    """
    Riemannian Conjugate Gradient on the sphere manifold.
    """
    start_time = time.time()
    current = _project_to_sphere(pts.copy())
    f, g = _energy_grad_vec(current.ravel())
    
    # Initial search direction: negative gradient
    d = -g.reshape(N, 3)
    # Ensure d is tangent
    d = d - np.sum(d * current, axis=1, keepdims=True) * current
    
    for k in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        if np.linalg.norm(g) < tol:
            break
        
        # Line search: minimize f along geodesic
        # Geodesic: x(t) = cos(t)*x0 + sin(t)*d/||d||
        d_norm = np.linalg.norm(d)
        if d_norm < 1e-15:
            break
        d_unit = d / d_norm
        
        # Backtracking line search
        step = 0.5
        for ls_iter in range(30):
            t = step * d_norm  # angle
            new_pts = np.cos(t) * current + np.sin(t) * d_unit
            new_pts = _project_to_sphere(new_pts)
            new_f, new_g = _energy_grad_vec(new_pts.ravel())
            
            if new_f < f:
                # Check sufficient decrease
                if new_f <= f + 1e-4 * t * np.dot(g, d):
                    break
            step *= 0.5
        
        if not np.isfinite(new_f) or new_f >= f:
            # Try smaller step
            t = 0.01 * d_norm
            new_pts = np.cos(t) * current + np.sin(t) * d_unit
            new_pts = _project_to_sphere(new_pts)
            new_f, new_g = _energy_grad_vec(new_pts.ravel())
        
        # Update
        current = new_pts
        f = new_f
        
        # Fletcher-Reeves beta
        g_new = new_g.reshape(N, 3)
        g_new = g_new - np.sum(g_new * current, axis=1, keepdims=True) * current
        g_old = g
        beta = np.dot(g_new, g_new) / max(np.dot(g_old, g_old), 1e-15)
        beta = min(beta, 0.9)  # Damp beta
        
        d = -g_new + beta * d
        # Project d to tangent
        d = d - np.sum(d * current, axis=1, keepdims=True) * current
        
        g = g_new
    
    return _project_to_sphere(current), f

def _polish(pts, max_iter=100, tol=1e-12, time_limit=10.0):
    """High-precision polish using L-BFGS-B with tight tolerances."""
    flat = pts.ravel()
    start_time = time.time()
    
    result = minimize(
        _energy_grad_vec,
        flat,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'maxfun': max_iter * 2, 
                 'ftol': tol, 'gtol': tol, 'maxls': 20}
    )
    
    opt_pts = _project_to_sphere(result.x.reshape(N, 3))
    return opt_pts, result.fun

def _fibonacci_spiral(n, offset=0.5):
    """Generate n points using Fibonacci spiral with optional offset."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _golden_spiral(n, offset=0.0):
    """Golden ratio spiral variant."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = 2.0 * np.pi * i / (1.0 + np.sqrt(5.0))
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, seed=None):
    """Generate random points on sphere."""
    if seed is not None:
        np.random.seed(seed)
    pts = np.random.randn(n, 3)
    return _project_to_sphere(pts)

def _perturb_and_escape(pts, base_energy, n_perturb=50, time_limit=15.0):
    """
    Try to escape local minima by perturbing and re-optimizing.
    """
    start_time = time.time()
    best_pts = pts.copy()
    best_energy = base_energy
    
    for i in range(n_perturb):
        if time.time() - start_time > time_limit:
            break
        
        # Random perturbation
        noise_scale = 0.05 + 0.05 * (i / n_perturb)
        noise = np.random.randn(N, 3) * noise_scale
        perturbed = _project_to_sphere(pts + noise)
        
        # Quick CG optimization
        opt_pts, opt_energy = _riemannian_cg(perturbed, max_iter=50, time_limit=2.0)
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_pts = opt_pts.copy()
    
    return best_pts, best_energy

def _time_left(deadline):
    return max(0.0, deadline - time.time())

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_energy = float('inf')
    best_points = None
    
    # Generate diverse starting configurations
    starts = []
    
    # Fibonacci spirals with different offsets
    for offset in [0.5, 0.1, 0.9]:
        starts.append(_fibonacci_spiral(N, offset))
    
    # Golden spiral
    starts.append(_golden_spiral(N, 0.0))
    
    # Random starts
    for seed in [42, 123, 456, 789, 1011]:
        starts.append(_random_points(N, seed))
    
    # Optimize each start with Riemannian CG
    for i, start_pts in enumerate(starts):
        if _time_left(deadline) < 10.0:
            break
        
        remaining = _time_left(deadline)
        # Allocate more time to early starts
        time_alloc = min(20.0, remaining * 0.3)
        
        opt_pts, opt_energy = _riemannian_cg(start_pts, max_iter=150, time_limit=time_alloc)
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_points = opt_pts.copy()
    
    # Polish the best result
    if best_points is not None and _time_left(deadline) > 5.0:
        polished, polished_energy = _polish(
            best_points, 
            max_iter=200, 
            time_limit=min(10.0, _time_left(deadline) - 2.0)
        )
        if polished_energy < best_energy:
            best_energy = polished_energy
            best_points = polished
    
    # Try to escape local minima if time permits
    if best_points is not None and _time_left(deadline) > 15.0:
        escaped, escaped_energy = _perturb_and_escape(
            best_points, 
            best_energy, 
            n_perturb=30,
            time_limit=min(25.0, _time_left(deadline) - 5.0)
        )
        if escaped_energy < best_energy:
            best_energy = escaped_energy
            best_points = escaped
            
            # Polish the escaped result
            if _time_left(deadline) > 5.0:
                polished, polished_energy = _polish(
                    best_points,
                    max_iter=100,
                    time_limit=min(8.0, _time_left(deadline) - 2.0)
                )
                if polished_energy < best_energy:
                    best_energy = polished_energy
                    best_points = polished
    
    # Final validation and normalization
    if best_points is None:
        best_points = _fibonacci_spiral(N, 0.5)
    
    best_points = _project_to_sphere(best_points)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                # Nudge point j slightly
                noise = np.random.randn(3) * 1e-6
                best_points[j] = _project_to_sphere(best_points[j] + noise)
    
    return best_points
