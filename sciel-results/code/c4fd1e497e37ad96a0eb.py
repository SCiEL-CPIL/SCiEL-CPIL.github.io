"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import time

# EVOLVE-BLOCK-START
import numpy as np
import math
from scipy.optimize import minimize, Bounds

def _fibonacci_spiral(n):
    """Generate n points on the unit sphere using the golden-angle spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _project_to_sphere(pts):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _log_energy_and_grad(pts):
    """
    Compute the logarithmic energy and its gradient on the sphere.
    
    Energy: sum_{i<j} log(1 / |x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    
    For points on the unit sphere, the gradient of the energy with respect to
    point x_i is:
        grad_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    But we need to project this to the tangent space (remove the radial component).
    Actually, since we're optimizing on the sphere, we can just compute the gradient
    in R^3 and use it directly with the constraint that points stay on the sphere.
    
    Let's be more careful. The energy is:
        E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    
    The partial derivative with respect to x_i:
        dE/dx_i = -sum_{j != i} d/dx_i log(|x_i - x_j|)
                = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    
    This is the gradient in R^3. When optimizing on the sphere, we project to the
    tangent space: grad_tangent_i = grad_i - (grad_i · x_i) * x_i
    """
    n = len(pts)
    # Compute pairwise distances
    # pts has shape (n, 3)
    # diff[i,j] = pts[i] - pts[j]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dists = np.linalg.norm(diff, axis=2)  # (n, n)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dists, 1.0)
    
    # Energy: sum over i<j of log(1/d) = -sum over i<j of log(d)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dists[iu]))
    
    # Gradient: for each i, grad_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # diff[i,j] = x_i - x_j
    # We want sum_j (diff[i,j] / dists[i,j]^2) for j != i
    # diff / dists^2 has shape (n, n, 3)
    inv_d2 = 1.0 / (dists ** 2)
    inv_d2[np.diag_indices(n)] = 0.0  # zero out diagonal
    
    # grad_i = -sum_j diff[i,j] * inv_d2[i,j]
    grad = -np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)  # (n, 3)
    
    # Project to tangent space
    # grad_tangent_i = grad_i - (grad_i · x_i) * x_i
    dot = np.sum(grad * pts, axis=1, keepdims=True)  # (n, 1)
    grad_tangent = grad - dot * pts  # (n, 3)
    
    return energy, grad_tangent

def _log_energy(pts):
    """Compute just the logarithmic energy."""
    energy, _ = _log_energy_and_grad(pts)
    return energy

def construct():
    """
    Minimize the logarithmic energy of 100 points on the unit sphere.
    Uses a multi-stage approach:
    1. Start with Fibonacci spiral
    2. Use gradient-based optimization with sphere constraint
    3. Multiple restarts with different initializations
    """
    deadline = time.time() + 140.0  # 10s safety margin
    best_energy = np.inf
    best_pts = _fibonacci_spiral(N)
    
    # Reshape to flat vector for scipy.optimize
    def energy_flat(flat_pts):
        pts = flat_pts.reshape(N, 3)
        pts = _project_to_sphere(pts)
        e, _ = _log_energy_and_grad(pts)
        return e
    
    def grad_flat(flat_pts):
        pts = flat_pts.reshape(N, 3)
        pts = _project_to_sphere(pts)
        _, grad_t = _log_energy_and_grad(pts)
        return grad_t.reshape(-1)
    
    # Try multiple initializations
    initializations = []
    
    # 1. Fibonacci spiral
    initializations.append(_fibonacci_spiral(N))
    
    # 2. Random initializations
    rng = np.random.default_rng(42)
    for _ in range(3):
        pts = rng.standard_normal((N, 3))
        initializations.append(_project_to_sphere(pts))
    
    # 3. Perturbed Fibonacci
    pts_pert = _fibonacci_spiral(N) + 0.1 * rng.standard_normal((N, 3))
    initializations.append(_project_to_sphere(pts_pert))
    
    for init_pts in initializations:
        if time.time() > deadline - 5.0:
            break
        
        flat_init = init_pts.reshape(-1)
        
        try:
            result = minimize(
                fun=energy_flat,
                x0=flat_init,
                jac=grad_flat,
                method='L-BFGS-B',
                options={
                    'maxiter': 2000,
                    'ftol': 1e-14,
                    'gtol': 1e-12,
                    'maxls': 50,
                    'maxfun': 5000,
                }
            )
            
            if result.success or result.fun < best_energy:
                best_pts = _project_to_sphere(result.x.reshape(N, 3))
                best_energy = result.fun
                
        except Exception:
            continue
    
    # Final polish: run more iterations from the best point
    if time.time() < deadline - 10.0:
        flat_best = best_pts.reshape(-1)
        try:
            result = minimize(
                fun=energy_flat,
                x0=flat_best,
                jac=grad_flat,
                method='L-BFGS-B',
                options={
                    'maxiter': 5000,
                    'ftol': 1e-15,
                    'gtol': 1e-13,
                    'maxls': 100,
                    'maxfun': 10000,
                }
            )
            if result.fun < best_energy:
                best_pts = _project_to_sphere(result.x.reshape(N, 3))
                best_energy = result.fun
        except Exception:
            pass
    
    # Ensure the result is valid
    best_pts = _project_to_sphere(best_pts)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Perturb slightly
                best_pts[j] += 1e-6 * np.random.randn(3)
                best_pts[j] = _project_to_sphere(best_pts[j])
    
    return best_pts
# EVOLVE-BLOCK-END
