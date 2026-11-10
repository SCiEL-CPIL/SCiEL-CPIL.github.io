"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
from math import log, fsum
import time
from scipy.optimize import minimize

DEADLINE = 120.0
SAFETY_MARGIN = 0.92  # use 92% of time

def _log_energy_precise(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) using fsum."""
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return 1e30
            terms.append(-log(d))
    return fsum(terms)

def _fibonacci_spiral(n, offset=0.0):
    """Golden-angle spiral points on unit sphere with optional phase offset."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, rng):
    """Random points on unit sphere."""
    v = rng.standard_normal((n, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms

def _energy_and_grad_cartesian(pts):
    """
    Compute logarithmic energy and its gradient w.r.t. Cartesian coordinates.
    Energy = sum_{i<j} -log(|x_i - x_j|)
    grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    # Pairwise differences: diff[i,j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    # Squared distances
    r2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(r2, 1.0)
    
    # Inverse squared distances
    inv_r2 = 1.0 / r2
    np.fill_diagonal(inv_r2, 0.0)
    
    # Gradient: grad_i = sum_j (x_i - x_j) / |x_i - x_j|^2
    grad = np.sum(diff * inv_r2[:, :, np.newaxis], axis=1)
    
    # Energy: sum_{i<j} -log(|x_i - x_j|) = -0.5 * sum_{i!=j} -log(|x_i - x_j|)
    r = np.sqrt(r2)
    np.fill_diagonal(r, 1.0)
    log_r = np.log(r)
    np.fill_diagonal(log_r, 0.0)
    energy = 0.5 * np.sum(log_r)
    
    return energy, grad

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _projected_gradient_descent(pts, n_iters=500, lr=None, deadline=None):
    """
    Gradient descent with projection onto sphere.
    Uses adaptive learning rate.
    """
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    best_energy = _log_energy_precise(pts)
    
    # Adaptive learning rate schedule
    base_lr = 0.05 if lr is None else lr
    
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        energy, grad = _energy_and_grad_cartesian(pts)
        
        # Project gradient to tangent space: g_tan = g - (g . x) * x
        grad_dot_x = np.sum(grad * pts, axis=1, keepdims=True)
        grad_tan = grad - grad_dot_x * pts
        
        # Adaptive step size: decrease if energy increases
        step = base_lr / (1.0 + it * 0.02)
        
        new_pts = pts - step * grad_tan
        new_pts = _project_to_sphere(new_pts)
        
        # Check minimum distance
        diff = new_pts[:, np.newaxis, :] - new_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist < 1e-10:
            # Too close, reduce step
            step *= 0.5
            new_pts = pts - step * grad_tan
            new_pts = _project_to_sphere(new_pts)
        
        # Accept if energy decreased
        new_energy = _log_energy_precise(new_pts)
        if new_energy < best_energy:
            pts = new_pts
            best_energy = new_energy
        else:
            # Halve step size
            base_lr *= 0.95
    
    return pts

def _newton_polish(pts, max_iters=50, deadline=None):
    """
    Newton-type polish on the sphere using the Hessian approximation.
    For the logarithmic energy, we use a Gauss-Newton-like approach.
    """
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    best_energy = _log_energy_precise(pts)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        energy, grad = _energy_and_grad_cartesian(pts)
        
        # Project gradient to tangent space
        grad_dot_x = np.sum(grad * pts, axis=1, keepdims=True)
        grad_tan = grad - grad_dot_x * pts
        
        # Check convergence
        grad_norm = np.linalg.norm(grad_tan)
        if grad_norm < 1e-12:
            break
        
        # Approximate Hessian in tangent space using finite differences
        # For efficiency, use a simple diagonal approximation
        # H_ii ≈ ||grad_i||^2 / |x_i|^2 (crude but effective for polish)
        # Better: use the fact that for log energy, the Hessian involves
        # second derivatives of log distances
        
        # Use a line search Newton step
        # Step direction: -H^{-1} g in tangent space
        # Approximate H as identity scaled by typical curvature
        
        # Compute a better Hessian approximation
        # For pair (i,j): contribution to H_ii from j is (1/r^2 - 3*(diff_ij . x_i)^2/r^4) * I - ...
        # This is complex, so use a damped Newton with backtracking
        
        # Simple approach: use gradient descent with small steps for polish
        step_size = 1.0 / (grad_norm + 1e-10)
        step_size = min(step_size, 0.1)
        
        new_pts = pts - step_size * grad_tan
        new_pts = _project_to_sphere(new_pts)
        
        # Backtracking line search
        alpha = 1.0
        for _ in range(20):
            trial = pts - alpha * step_size * grad_tan
            trial = _project_to_sphere(trial)
            
            # Check validity
            diff = trial[:, np.newaxis, :] - trial[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff * diff, axis=2))
            np.fill_diagonal(dists, np.inf)
            if np.min(dists) < 1e-10:
                alpha *= 0.5
                continue
            
            trial_energy = _log_energy_precise(trial)
            if trial_energy < best_energy:
                pts = trial
                best_energy = trial_energy
                break
            alpha *= 0.5
        else:
            break  # No improvement found
    
    return pts

def _repulsion_refine_cartesian(pts, n_iters=200, lr=0.02, deadline=None):
    """
    Iterative repulsion in Cartesian space to spread points.
    Pre-conditioner before main optimization.
    """
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    for it in range(n_iters):
        if deadline is not None and time.time() > deadline:
            break
        # Vectorized force calculation
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d2, 1.0)
        inv_d2 = 1.0 / d2
        np.fill_diagonal(inv_d2, 0.0)
        forces = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
        
        # Adaptive learning rate
        eff_lr = lr / (1.0 + it * 0.05)
        pts += eff_lr * forces
        pts = _project_to_sphere(pts)
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE * SAFETY_MARGIN
    
    rng = np.random.default_rng(42)
    
    # Generate multiple initial configurations
    inits = []
    # Fibonacci spiral variants
    inits.append(_fibonacci_spiral(N))
    inits.append(_fibonacci_spiral(N, offset=0.5))
    inits.append(_fibonacci_spiral(N, offset=1.0))
    inits.append(_fibonacci_spiral(N, offset=2.0))
    # Random starts
    for _ in range(4):
        inits.append(_random_points(N, rng))
    
    best_pts = None
    best_energy = 1e30
    
    # For each initialization:
    # 1. Pre-condition with Cartesian repulsion
    # 2. Gradient descent on sphere
    # 3. Newton polish
    for init_idx, init in enumerate(inits):
        if time.time() > deadline - 10:  # Leave time for final polish
            break
            
        # 1. Pre-condition
        refined_cart = _repulsion_refine_cartesian(init, n_iters=200, lr=0.02, deadline=deadline)
        
        # 2. Gradient descent on sphere
        gd_pts = _projected_gradient_descent(refined_cart, n_iters=300, lr=0.05, deadline=deadline)
        
        # 3. Newton polish
        polished_pts = _newton_polish(gd_pts, max_iters=30, deadline=deadline)
        
        # Check validity
        diff = polished_pts[:, np.newaxis, :] - polished_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist < 1e-9:
            continue
        
        # Calculate precise energy
        e = _log_energy_precise(polished_pts)
        
        if e < best_energy:
            best_energy = e
            best_pts = polished_pts.copy()
    
    # Final Polish Phase on the best solution
    if best_pts is not None and time.time() < deadline - 2:
        # Extra high-precision polish
        final_pts = _newton_polish(best_pts, max_iters=50, deadline=deadline)
        
        diff = final_pts[:, np.newaxis, :] - final_pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        
        if min_dist >= 1e-9:
            e_final = _log_energy_precise(final_pts)
            if e_final < best_energy:
                best_energy = e_final
                best_pts = final_pts.copy()
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
    
    # Ensure unit norm
    best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
