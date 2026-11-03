import numpy as np
import time
import math
from scipy.optimize import minimize

N = 150
DEADLINE = 115.0  # Safety margin from 120s

def _log_energy_fast(pts):
    """
    Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|).
    Returns (energy, gradient) for optimization.
    """
    n = len(pts)
    # Ensure points are on sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    
    dot = pts @ pts.T
    dist_sq = np.maximum(2.0 - 2.0 * dot, 1e-15)
    np.fill_diagonal(dist_sq, 1.0)
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    if np.any(d_sq < 1e-18):
        return float('inf'), None
    terms = -0.5 * np.log(d_sq)
    energy = math.fsum(terms.tolist())
    
    # Gradient calculation for the normalized points (tangent space)
    # g_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist_sq_full = np.sum(diff * diff, axis=2)  # (N, N)
    np.fill_diagonal(dist_sq_full, 1.0)
    weights = 1.0 / dist_sq_full
    np.fill_diagonal(weights, 0.0)
    grad_raw = np.einsum('ij,ijk->ik', weights, diff)
    
    # Project to tangent space
    proj_factor = np.sum(grad_raw * pts, axis=1, keepdims=True)
    grad_tangent = grad_raw - proj_factor * pts
    
    return energy, grad_tangent

def _objective_and_grad(y_flat):
    """
    Objective function for L-BFGS-B.
    Input: y_flat (3N,)
    Output: energy, gradient (3N,)
    """
    n = len(y_flat) // 3
    y = y_flat.reshape(n, 3)
    
    # Normalize to sphere
    norms = np.linalg.norm(y, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.maximum(norms, 1e-15)
    pts = y / norms
    
    # Compute energy and tangent gradient
    energy, grad_tangent = _log_energy_fast(pts)
    
    if energy == float('inf'):
        return energy, np.zeros_like(y_flat)
        
    # Map tangent gradient back to raw space gradient
    # dE/dy_i = grad_tangent_i / ||y_i||
    grad_raw = grad_tangent / norms
    
    return energy, grad_raw.flatten()

def _fibonacci_sphere(n):
    """Golden-angle Fibonacci spiral points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_uniform_sphere(n, seed):
    np.random.seed(seed)
    theta = np.random.uniform(0, 2 * np.pi, n)
    phi = np.arccos(2 * np.random.uniform(0, 1, n) - 1)
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def construct():
    global DEADLINE
    start_time = time.time()
    
    # Initial candidate: Fibonacci sphere
    best_pts = _fibonacci_sphere(N)
    best_energy, _ = _log_energy_fast(best_pts)
    
    # Simulated Annealing
    # We use a cooling schedule that allows many steps
    # T starts high, decreases linearly or exponentially
    T_start = 10.0
    T_end = 1e-4
    steps = 50000  # Tentative, will be limited by time
    
    current_pts = best_pts.copy()
    current_energy = best_energy
    
    # Pre-calculate some constants if needed
    # For SA, we need to evaluate energy of a perturbed point
    # To speed this up, we can't easily do it incrementally without complex bookkeeping,
    # so we rely on the vectorized _log_energy_fast being fast enough.
    # N=150 -> 150x150 matrix ops. Should be ~1ms.
    # 50k steps * 1ms = 50s. Feasible.
    
    rng = np.random.default_rng(42)
    
    for i in range(steps):
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 15:  # Leave 15s for polish
            break
            
        # Linear cooling
        T = T_start * (T_end / T_start) ** (i / steps)
        
        # Perturb: pick a random point, add Gaussian noise, project to sphere
        idx = rng.integers(0, N)
        noise = rng.normal(0, 0.01, size=3)
        new_pts = current_pts.copy()
        new_pts[idx] = current_pts[idx] + noise
        new_pts = _project_to_sphere(new_pts)
        
        new_energy, _ = _log_energy_fast(new_pts)
        
        if not np.isfinite(new_energy):
            continue
            
        delta_E = new_energy - current_energy
        
        # Metropolis criterion
        if delta_E < 0:
            current_pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = current_pts.copy()
        else:
            # Accept with probability exp(-delta_E / T)
            if rng.random() < math.exp(-delta_E / T):
                current_pts = new_pts
                current_energy = new_energy
    
    # Final Polish with L-BFGS-B
    if time.time() - start_time < DEADLINE - 5:
        try:
            init_y = best_pts.flatten()
            res = minimize(
                _objective_and_grad,
                init_y,
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 200,
                    'maxfun': 200,
                    'ftol': 1e-10,
                    'gtol': 1e-8,
                    'disp': False
                }
            )
            y_opt = res.x.reshape(N, 3)
            norms = np.linalg.norm(y_opt, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            pts_opt = y_opt / norms
            
            e, _ = _log_energy_fast(pts_opt)
            if e < best_energy and np.isfinite(e):
                best_energy = e
                best_pts = pts_opt
        except Exception:
            pass

    # Fallback
    if best_pts is None:
        best_pts = _fibonacci_sphere(N)

    # Final normalization and check
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    best_pts = best_pts / norms
    
    return best_pts
