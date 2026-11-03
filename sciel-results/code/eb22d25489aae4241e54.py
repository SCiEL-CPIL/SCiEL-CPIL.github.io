"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 100

# EVOLVE-BLOCK-START

def _energy_and_tangent_grad(pts):
    """
    Compute logarithmic energy and the Riemannian gradient on the sphere.
    
    Energy E = sum_{i<j} log(1/||x_i - x_j||)
    
    For points on the unit sphere, the Riemannian gradient of the energy w.r.t. x_i
    is the projection of the Euclidean gradient onto the tangent space at x_i.
    
    Euclidean gradient w.r.t. x_i:
    grad_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    
    Wait, let's re-verify the sign.
    f(x) = -log(||x_i - x_j||)
    d/dx_i (-log(||x_i - x_j||)) = -1/||x_i - x_j|| * (x_i - x_j)/||x_i - x_j|| = -(x_i - x_j)/||x_i - x_j||^2
    So Euclidean grad_i = sum_{j != i} -(x_i - x_j) / ||x_i - x_j||^2 = sum_{j != i} (x_j - x_i) / ||x_i - x_j||^2
    
    Riemannian gradient = Euclidean grad - <Euclidean grad, x_i> x_i
    Since x_i is on the sphere, <x_i, x_i> = 1.
    Note that <x_j - x_i, x_i> = <x_j, x_i> - 1.
    
    So Riemannian grad_i = sum_{j != i} (x_j - x_i) / d_ij^2 - x_i * sum_{j != i} (x_j - x_i) / d_ij^2 . x_i
    
    Actually, a simpler way to compute the Riemannian gradient is:
    Rgrad_i = (I - x_i x_i^T) * Egrad_i
    
    Let's compute Egrad first.
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    mask = np.ones((n, n), dtype=bool)
    np.fill_diagonal(mask, False)
    
    if np.any(dist_sq[mask] < 1e-18):
        return float('inf'), np.zeros_like(pts)
        
    dists = np.sqrt(dist_sq)
    i, j = np.triu_indices(n, k=1)
    d_upper = dists[i, j]
    
    if np.any(d_upper < 1e-9):
        return float('inf'), np.zeros_like(pts)
        
    energy = -np.sum(np.log(d_upper))
    
    inv_dist_sq = np.zeros_like(dist_sq)
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    
    # Egrad_i = sum_{j != i} (x_j - x_i) / d_ij^2 = - sum_{j != i} (x_i - x_j) / d_ij^2
    # diff[i,j] = x_i - x_j
    # term = -diff[i,j] * inv_dist_sq[i,j]
    # Egrad = sum_j term
    G = -diff * inv_dist_sq[:, :, np.newaxis]
    egrad = np.sum(G, axis=1)
    
    # Project to tangent space: Rgrad = egrad - <egrad, x> x
    # <egrad, x> is a vector of dot products
    dot = np.sum(egrad * pts, axis=1)
    rgrad = egrad - dot[:, np.newaxis] * pts
    
    return energy, rgrad

def _exp_map(pts, delta):
    """
    Exponential map on the sphere.
    For a point x on the sphere and a tangent vector v (||v|| < pi),
    exp_x(v) = cos(||v||) x + sin(||v||) / ||v|| v
    
    If ||v|| is small, use Taylor expansion to avoid numerical issues.
    """
    norms = np.linalg.norm(delta, axis=1, keepdims=True)
    # Avoid division by zero
    safe_norms = np.where(norms < 1e-8, 1.0, norms)
    
    # Handle small norms with Taylor expansion:
    # cos(t) ~ 1 - t^2/2
    # sin(t)/t ~ 1 - t^2/6
    cos_t = np.cos(norms)
    sin_t_over_t = np.sin(norms) / safe_norms
    
    # For very small norms, sin(t)/t is 1.0
    small_mask = norms < 1e-4
    sin_t_over_t[small_mask] = 1.0 - (norms[small_mask] ** 2) / 6.0
    cos_t[small_mask] = 1.0 - (norms[small_mask] ** 2) / 2.0
    
    new_pts = cos_t * pts + sin_t_over_t * delta
    # Normalize to ensure we stay on the sphere (corrects for floating point errors)
    new_norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    new_norms = np.where(new_norms < 1e-12, 1.0, new_norms)
    return new_pts / new_norms

def _local_minimize_manifold(start_pts, deadline, max_iter=100):
    """
    Perform Riemannian Gradient Descent with adaptive step size (Backtracking).
    """
    pts = start_pts.copy()
    current_energy, current_grad = _energy_and_tangent_grad(pts)
    
    if not np.isfinite(current_energy):
        return pts
    
    step_size = 1.0
    min_step_size = 1e-8
    
    for _ in range(max_iter):
        if time.time() > deadline:
            break
            
        # Backtracking line search
        alpha = step_size
        improved = False
        
        while alpha > min_step_size:
            delta = -alpha * current_grad
            new_pts = _exp_map(pts, delta)
            new_energy, new_grad = _energy_and_tangent_grad(new_pts)
            
            if np.isfinite(new_energy) and new_energy < current_energy:
                pts = new_pts
                current_energy = new_energy
                current_grad = new_grad
                improved = True
                # Increase step size if we improved significantly? 
                # Or keep it. Standard backtracking keeps it or increases slightly.
                # Let's keep it simple: don't increase, just don't decrease next time if we didn't shrink.
                break
            else:
                alpha *= 0.5
        
        if not improved:
            # If we couldn't find a step, try a random perturbation to escape local minimum?
            # Or just break.
            # Let's try a small random perturbation in tangent space
            noise = np.random.normal(scale=0.01, size=(N, 3))
            # Project noise to tangent space
            dot = np.sum(noise * pts, axis=1)
            tangent_noise = noise - dot[:, np.newaxis] * pts
            new_pts = _exp_map(pts, tangent_noise)
            new_energy, new_grad = _energy_and_tangent_grad(new_pts)
            if np.isfinite(new_energy) and new_energy < current_energy:
                pts = new_pts
                current_energy = new_energy
                current_grad = new_grad
            else:
                break
                
        # Check convergence
        if np.linalg.norm(current_grad) < 1e-8:
            break
            
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin for 120s limit
    
    best_pts = None
    best_energy = float('inf')
    
    # 1. Fibonacci spiral initialization
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Optimize Fibonacci
    opt_fib = _local_minimize_manifold(fib_pts, deadline)
    e_fib, _ = _energy_and_tangent_grad(opt_fib)
    if e_fib < best_energy:
        best_energy = e_fib
        best_pts = opt_fib.copy()
        
    if time.time() > deadline:
        return best_pts

    rng = np.random.default_rng(42)
    
    # 2. Multi-start with random and perturbed starts
    # We can afford more iterations per start if they are faster, but RGD is iterative.
    # Let's do 30 restarts.
    
    for k in range(30):
        if time.time() > deadline:
            break
            
        if k % 2 == 0 and best_pts is not None:
            # Perturb best solution
            # Generate random tangent vectors
            noise = rng.normal(size=(N, 3))
            dot = np.sum(noise * best_pts, axis=1)
            tangent_noise = noise - dot[:, np.newaxis] * best_pts
            # Scale noise
            scale = 0.5 + 0.5 * (k % 10) / 10.0 # Varying scale
            delta = scale * tangent_noise
            pts = _exp_map(best_pts, delta)
        else:
            # Random start
            pts = rng.normal(size=(N, 3))
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            pts = pts / norms
            
        opt_pts = _local_minimize_manifold(pts, deadline)
        
        e_opt, _ = _energy_and_tangent_grad(opt_pts)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = opt_pts.copy()
            
    # Final check
    if best_pts is None:
        best_pts = fib_pts
    else:
        norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        best_pts = best_pts / norms
        
    return best_pts
# EVOLVE-BLOCK-END
