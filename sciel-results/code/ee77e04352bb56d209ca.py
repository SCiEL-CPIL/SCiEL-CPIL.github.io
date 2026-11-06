"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
import math

N = 150

# EVOLVE-BLOCK-START

def _energy_and_grad(pts):
    """
    Compute logarithmic energy and its gradient for points on the unit sphere.
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient with respect to x_i is the sum over j!=i of (x_i - x_j)/|x_i - x_j|^2
    (the gradient of -log(|x_i - x_j|) w.r.t x_i is (x_i - x_j)/|x_i - x_j|^2)
    
    For minimization of energy, we want to minimize E = -sum log(d_ij)
    dE/dx_i = -sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2 ... wait let me re-derive.
    
    E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    
    d/dx_i [ -log(|x_i - x_j|) ] = -(x_i - x_j)/|x_i - x_j|^2
    
    So grad_i E = -sum_{j!=i} (x_i - x_j)/|x_i - x_j|^2 = sum_{j!=i} (x_j - x_i)/|x_i - x_j|^2
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    
    # Compute all pairwise distances
    # diff[i,j] = x_i - x_j
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # shape (n, n, 3)
    dist = np.linalg.norm(diff, axis=2)  # shape (n, n)
    
    # Avoid division by zero (diagonal)
    np.fill_diagonal(dist, 1.0)
    
    # Energy
    # E = -sum_{i<j} log(d_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient
    # grad_i = sum_{j!=i} (x_j - x_i) / |x_i - x_j|^2
    # diff[i,j] = x_i - x_j, so x_j - x_i = -diff[i,j]
    # grad_i = -sum_{j!=i} diff[i,j] / dist[i,j]^2
    inv_dist_sq = 1.0 / (dist * dist)  # shape (n, n)
    # Set diagonal to 0
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _lbfgsb_optimize(pts0, maxiter=2000, tol=1e-12, time_limit=None, start_time=None):
    """
    Run L-BFGS-B optimization with a time limit.
    Returns (best_pts, best_energy, converged).
    """
    if start_time is None:
        start_time = time.time()
    
    best_pts = pts0.copy()
    best_energy = _energy_and_grad(pts0)[0]
    converged = False
    
    def objective(x):
        nonlocal best_pts, best_energy
        pts = _project_to_sphere(x)
        energy, grad = _energy_and_grad(pts)
        
        # Update best
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Time check
        if time_limit is not None and (time.time() - start_time) > time_limit:
            raise TimeoutError("Time limit exceeded")
        
        return energy, grad
    
    try:
        result = minimize(
            objective,
            pts0,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'ftol': tol, 'gtol': 1e-10, 'maxls': 50}
        )
        converged = result.success
    except TimeoutError:
        pass
    except Exception:
        pass
    
    # Final projection and energy check
    best_pts = _project_to_sphere(best_pts)
    best_energy = _energy_and_grad(best_pts)[0]
    
    return best_pts, best_energy, converged


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 150s limit, 5s safety margin
    
    # Start with Fibonacci spiral
    pts = _fibonacci_sphere(N)
    best_pts = pts.copy()
    best_energy = _energy_and_grad(pts)[0]
    
    # Try multiple restarts with different initial configurations
    np.random.seed(42)
    
    restarts = []
    restarts.append(("fib", _fibonacci_sphere(N)))
    
    # Add some random restarts
    for _ in range(5):
        random_pts = np.random.randn(N, 3)
        restarts.append(("rand", _project_to_sphere(random_pts)))
    
    # Also try slightly perturbed Fibonacci
    for _ in range(3):
        perturbed = _fibonacci_sphere(N) + 0.01 * np.random.randn(N, 3)
        restarts.append(("perturbed", _project_to_sphere(perturbed)))
    
    for name, init_pts in restarts:
        if time.time() > deadline - 30.0:
            break
        
        remaining_time = deadline - time.time() - 5.0
        if remaining_time < 5.0:
            break
        
        pts_opt, energy_opt, conv = _lbfgsb_optimize(
            init_pts,
            maxiter=1000,
            tol=1e-12,
            time_limit=remaining_time,
            start_time=start_time
        )
        
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt.copy()
    
    # Final polish with tighter tolerance if time permits
    if time.time() < deadline - 10.0:
        remaining_time = deadline - time.time() - 5.0
        if remaining_time > 5.0:
            pts_final, energy_final, _ = _lbfgsb_optimize(
                best_pts,
                maxiter=500,
                tol=1e-14,
                time_limit=remaining_time,
                start_time=start_time
            )
            if energy_final < best_energy:
                best_energy = energy_final
                best_pts = pts_final.copy()
    
    # Ensure points are on the unit sphere and valid
    best_pts = _project_to_sphere(best_pts)
    
    # Final validation
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-15) or not np.all(np.isfinite(best_pts)):
        # Fallback to Fibonacci
        best_pts = _fibonacci_sphere(N)
    
    return best_pts

# EVOLVE-BLOCK-END
