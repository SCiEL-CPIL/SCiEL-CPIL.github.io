"""Rewritten program using L-BFGS-B with manifold gradient handling."""
import numpy as np
import math
import time
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

def _energy_and_tangent_grad(pts):
    """
    Compute logarithmic energy and tangent gradient on the sphere.
    pts: (N, 3) array of points assumed to be on the unit sphere (or normalized inside).
    Returns:
        energy: scalar
        grad_tan: (N, 3) array of gradients projected to the tangent space.
        u: (N, 3) normalized points
    """
    # Ensure points are normalized for calculation
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    u = pts / norms
    
    # Pairwise differences
    diff = u[:, None, :] - u[None, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Clip to avoid log(0) and division by 0
    dist_sq = np.clip(dist_sq, 1e-12, None)
    
    # Energy calculation
    # E = sum_{i<j} log(1/d) = -0.25 * sum_{i!=j} log(d_sq_ij)
    log_dist_sq = np.log(dist_sq)
    np.fill_diagonal(log_dist_sq, 0.0)
    energy = -0.25 * np.sum(log_dist_sq)
    
    # Gradient calculation
    # dE/du_i = sum_{j!=i} (u_j - u_i) / ||u_i - u_j||^2
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    w_sum = np.sum(inv_dist_sq, axis=1)
    
    term1 = inv_dist_sq @ u
    term2 = u * w_sum[:, None]
    
    grad_raw = term1 - term2
    
    # Project to tangent space
    dot = np.sum(grad_raw * u, axis=1)
    grad_tan = grad_raw - dot[:, None] * u
    
    return energy, grad_tan, u

def _objective_and_grad(x_flat):
    """
    Objective function and gradient for scipy.minimize.
    x_flat: (3N,) vector of unnormalized coordinates.
    """
    x = x_flat.reshape(N, 3)
    
    energy, grad_tan, u = _energy_and_tangent_grad(x)
    
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    
    grad_x = grad_tan / norms
    
    return energy, grad_x.flatten()

def _initial_fibonacci():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _initial_random(rng):
    """Random points on the unit sphere."""
    pts = rng.normal(size=(N, 3))
    return _project_to_sphere(pts)

def _initial_antipodal(rng):
    """Antipodal symmetric points."""
    pts_half = rng.normal(size=(N//2, 3))
    pts_full = np.vstack([pts_half, -pts_half])
    return _project_to_sphere(pts_full)

def _optimize_lbfgs(start_pts, deadline, is_polish=False, maxiter=None):
    """
    Optimize using L-BFGS-B with time-based callback.
    """
    x0 = start_pts.flatten()
    
    if is_polish:
        ftol = 1e-16
        gtol = 1e-15
        if maxiter is None:
            maxiter = 5000
    else:
        ftol = 1e-12
        gtol = 1e-10
        if maxiter is None:
            maxiter = 3000
    
    class TimeOutCallback:
        def __init__(self, deadline):
            self.deadline = deadline
            self.timeout = False
            self.last_time = time.time()
            
        def __call__(self, xk, f, g):
            if time.time() > self.deadline:
                self.timeout = True
            return self.timeout
    
    cb = TimeOutCallback(deadline)
    
    options = {
        'maxiter': maxiter,
        'maxfun': 10 * maxiter,
        'ftol': ftol,
        'gtol': gtol,
        'iprint': 0
    }
    
    try:
        res = minimize(
            _objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            callback=cb,
            options=options
        )
        return res.x.reshape(N, 3)
    except Exception:
        return start_pts.copy()

def _local_search(pts, deadline, max_iter=2000, step_scale=1.0):
    """
    Simple gradient descent with adaptive step size in tangent space.
    Used as a final polish or to escape shallow local minima.
    """
    pts = _project_to_sphere(pts.copy())
    e_curr, grad_tan, u = _energy_and_tangent_grad(pts)
    
    # Estimate initial step size based on gradient magnitude
    g_norm = np.linalg.norm(grad_tan)
    if g_norm < 1e-15:
        return pts, e_curr
    
    # Initial step: aim for ~1% change in energy per step initially
    # E ~ -0.5 * sum log(d). dE ~ grad . dx.
    # Let's start with a small step.
    step = 0.1 * step_scale
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
        
        # Move in negative gradient direction (tangent)
        # Normalize gradient to unit length for consistent step control
        g_norm = np.linalg.norm(grad_tan)
        if g_norm < 1e-15:
            break
        grad_norm = grad_tan / g_norm
        
        # Try step
        new_pts = pts - step * grad_norm
        new_pts = _project_to_sphere(new_pts)
        
        e_new, grad_new, u_new = _energy_and_tangent_grad(new_pts)
        
        if e_new < e_curr:
            pts = new_pts
            e_curr = e_new
            grad_tan = grad_new
            # Increase step if improvement is good? Or keep same.
            # Let's try to increase slightly if improvement is significant
            # improvement = (e_curr - e_new)
            # If improvement is large relative to step, we can take bigger step?
            # For now, keep simple. Maybe decay step slowly?
            # Actually, L-BFGS is better. This is a backup.
            pass
        else:
            # Decrease step
            step *= 0.5
            if step < 1e-12:
                break
        
        # Optional: increase step if we made good progress
        # if (e_curr - e_new) > 0.1 * step * g_norm: 
        #     step *= 1.1
        #     if step > 0.5: step = 0.5
        
    return pts, e_curr

def _refine_with_symmetry(pts, deadline):
    """
    Try to refine by enforcing antipodal symmetry if it helps.
    Map points to closest antipodal pair? No, that's complex.
    Instead, just run optimization on the antipodal symmetric version.
    """
    # Sort points by z-coordinate?
    # This is heuristic.
    # Better: just generate a new antipodal init and optimize.
    return None

def construct():
    """
    Main entry point.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    # 1. Generate diverse initializations
    inits = []
    
    # Fibonacci
    inits.append(_initial_fibonacci())
    
    # Random
    for _ in range(10):
        inits.append(_initial_random(rng))
        
    # Antipodal
    for _ in range(5):
        inits.append(_initial_antipodal(rng))
        
    # Jittered Fibonacci
    fib = _initial_fibonacci()
    for _ in range(5):
        pts = fib + rng.normal(size=(N, 3)) * 0.5
        inits.append(_project_to_sphere(pts))
        
    best_pts = inits[0].copy()
    best_energy, _, _ = _energy_and_tangent_grad(best_pts)
    
    # 2. Optimize each initialization
    # We have ~145s.
    # L-BFGS takes ~1-2s per run usually for N=100.
    # Let's allocate time per restart.
    
    n_restarts = len(inits)
    time_per_restart = 12.0  # seconds
    
    for i, init_pts in enumerate(inits):
        if time.time() > deadline - 20:  # Reserve 20s for final polish
            break
            
        t_start = time.time()
        t_deadline = t_start + time_per_restart
        
        try:
            optimized_pts = _optimize_lbfgs(init_pts, t_deadline, is_polish=False)
            optimized_pts = _project_to_sphere(optimized_pts)
            
            e_opt, _, _ = _energy_and_tangent_grad(optimized_pts)
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = optimized_pts.copy()
                
        except Exception:
            continue

    # 3. Final high-precision polish on the best found
    if time.time() < deadline - 10:
        try:
            # Run L-BFGS polish with tight tolerances
            t_deadline = deadline - 5
            final_pts = _optimize_lbfgs(best_pts, t_deadline, is_polish=True, maxiter=10000)
            final_pts = _project_to_sphere(final_pts)
            e_final, _, _ = _energy_and_tangent_grad(final_pts)
            
            if e_final < best_energy:
                best_energy = e_final
                best_pts = final_pts.copy()
                
        except Exception:
            pass
    
    # 4. Try a few more random restarts if time remains
    if time.time() < deadline - 15:
        for _ in range(5):
            if time.time() > deadline - 10:
                break
            try:
                init_pts = _initial_random(rng)
                t_deadline = min(deadline - 5, time.time() + 8.0)
                optimized_pts = _optimize_lbfgs(init_pts, t_deadline, is_polish=False)
                optimized_pts = _project_to_sphere(optimized_pts)
                e_opt, _, _ = _energy_and_tangent_grad(optimized_pts)
                
                if e_opt < best_energy:
                    best_energy = e_opt
                    best_pts = optimized_pts.copy()
            except Exception:
                continue

    # 5. Final validation
    best_pts = _project_to_sphere(best_pts)
    
    # Check for close points
    diff = best_pts[:, None, :] - best_pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(dist[iu])
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci if invalid
        best_pts = _initial_fibonacci()
    
    return best_pts

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms
# EVOLVE-BLOCK-END
