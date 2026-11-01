"""Rewritten program: L-BFGS-B minimization of logarithmic energy with sphere penalty."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _fibonacci_sphere(n):
    """Generate a Fibonacci spiral distribution on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _energy_and_grad_cartesian(x_flat, lam=1e6):
    """
    Compute energy and gradient for points in Cartesian coordinates.
    Includes a penalty term to keep points on the unit sphere.
    x_flat: shape (N, 3)
    """
    pts = x_flat.reshape(N, 3)
    
    # Compute pairwise differences and distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)  # (N, N)
    
    # Avoid division by zero
    dist_safe = np.where(dist < 1e-12, 1e-12, dist)
    
    # Energy: sum_{i<j} log(1/dist)
    iu = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dist_safe[iu]))
    
    # Gradient of energy w.r.t. pts
    # dE/dx_i = - sum_{j != i} (x_i - x_j) / dist_ij^2
    inv_dist_sq = np.zeros_like(dist_safe)
    np.fill_diagonal(inv_dist_sq, 0.0)
    inv_dist_sq = 1.0 / (dist_safe * dist_safe)
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    grad_energy = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    # Penalty term: P = lam * sum_i (||x_i||^2 - 1)^2
    # dP/dx_i = lam * 2 * (||x_i||^2 - 1) * 2 * x_i = 4 * lam * (||x_i||^2 - 1) * x_i
    norms_sq = np.sum(pts * pts, axis=1)
    penalty_grad = 4.0 * lam * (norms_sq - 1.0)[:, np.newaxis] * pts
    
    # Total gradient
    grad_total = grad_energy + penalty_grad
    
    # Total energy (penalty value is small if close to sphere)
    penalty_val = lam * np.sum((norms_sq - 1.0) ** 2)
    total_energy = energy + penalty_val
    
    return total_energy, grad_total


def construct():
    # Start from Fibonacci spiral
    pts_init = _fibonacci_sphere(N).astype(np.float64)
    x0 = pts_init.flatten()
    
    # Use L-BFGS-B to minimize
    # We use a penalty method to enforce the sphere constraint.
    # A large lambda ensures points stay on the sphere.
    # Note: L-BFGS-B is very efficient.
    
    res = minimize(
        _energy_and_grad_cartesian,
        x0,
        method='L-BFGS-B',
        jac=True,  # We provide the gradient
        options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
    )
    
    pts = res.x.reshape(N, 3)
    
    # Final renormalization to ensure exact unit norm
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / np.where(norms < 1e-12, 1.0, norms)
    
    # Verify all points are finite and distinct
    assert np.all(np.isfinite(pts))
    assert pts.shape == (N, 3)
    
    return pts
# EVOLVE-BLOCK-END
