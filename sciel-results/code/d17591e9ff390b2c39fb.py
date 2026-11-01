"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize


def _log_energy(pts):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    # Compute all pairwise distances
    # pts is (N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dists, 1.0)
    # We want sum over i<j of log(1/d) = -sum over i<j of log(d)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    d = dists[iu]
    return -np.sum(np.log(d))


def _log_energy_and_grad(pts):
    """Compute energy and gradient of the log energy on the sphere."""
    # pts: (N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    # Avoid division by zero on diagonal
    dists_sq += 1e-15
    dists = np.sqrt(dists_sq)
    # Gradient of E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    # dE/dx_k = sum_{j != k} (x_k - x_j) / |x_k - x_j|^2
    # This is because d/dx_k [-log(|x_k - x_j|)] = -(x_k - x_j)/|x_k - x_j|^2
    # Wait, E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # dE/dx_k = -sum_{j != k} d(log d_kj)/dx_k = -sum_{j != k} (x_k - x_j)/d_kj^2
    # But we want to MINIMIZE E, so gradient is as above.
    # Actually, let's re-derive.
    # E = - sum_{i<j} log(||x_i - x_j||)
    # dE/dx_k = - sum_{j != k} (1/||x_k - x_j||) * (x_k - x_j)/||x_k - x_j||
    #         = - sum_{j != k} (x_k - x_j)/||x_k - x_j||^2
    # So grad_k = - sum_{j != k} (x_k - x_j)/d_kj^2
    
    # Vectorized gradient computation
    # inv_d2 = 1 / dists_sq  (shape N x N)
    inv_d2 = 1.0 / dists_sq
    np.fill_diagonal(inv_d2, 0.0)
    # grad_k = - sum_j (x_k - x_j) * inv_d2[k,j]
    # diff[k,j] = x_k - x_j
    # grad = - sum_j diff[k,j] * inv_d2[k,j]  (over j, for each k)
    # grad = - (diff * inv_d2[:, :, np.newaxis]).sum(axis=1)
    grad = -(diff * inv_d2[:, :, np.newaxis]).sum(axis=1)
    
    # Energy
    iu = np.triu_indices(N, k=1)
    d = dists[iu]
    energy = -np.sum(np.log(d))
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _rstep(pts, step_size):
    """Riemannian step: move in the direction of the gradient on the sphere."""
    # pts: (N, 3)
    energy, grad = _log_energy_and_grad(pts)
    # Riemannian gradient = grad - (grad . x) x  (project onto tangent space)
    # For each point k: grad_tangent_k = grad_k - (grad_k . pts_k) * pts_k
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad_tangent = grad - dot * pts
    # Step
    new_pts = pts - step_size * grad_tangent
    # Project back to sphere
    new_pts = _project_to_sphere(new_pts)
    return new_pts, energy


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s
    
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_energy = _log_energy(pts)
    best_pts = pts.copy()
    
    # Riemannian gradient descent with adaptive step size
    step_size = 0.1
    min_step = 1e-8
    max_iter = 5000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        new_pts, energy = _rstep(pts, step_size)
        
        # Check for improvement
        if energy < best_energy:
            best_energy = energy
            best_pts = new_pts.copy()
            pts = new_pts
            # Increase step size slightly
            step_size = min(step_size * 1.05, 0.5)
        else:
            # Decrease step size
            step_size *= 0.8
            if step_size < min_step:
                break
    
    # Final polish with scipy minimize using L-BFGS-B on the sphere
    # We'll use a parameterization: each point is (theta, phi)
    # But it's easier to use Cartesian with a projection
    
    # Use scipy's minimize with a custom constraint approach
    # We'll minimize in Cartesian space with the constraint that |x|=1 for each point
    # This is tricky, so let's do a few more gradient descent steps with smaller step
    
    # Do a final high-precision polish
    step_size = 1e-4
    for it in range(2000):
        if time.time() > deadline:
            break
        
        energy, grad = _log_energy_and_grad(best_pts)
        dot = np.sum(grad * best_pts, axis=1, keepdims=True)
        grad_tangent = grad - dot * best_pts
        
        new_pts = best_pts - step_size * grad_tangent
        new_pts = _project_to_sphere(new_pts)
        
        new_energy = _log_energy(new_pts)
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = new_pts
        else:
            step_size *= 0.5
            if step_size < 1e-10:
                break
    
    return best_pts
# EVOLVE-BLOCK-END
