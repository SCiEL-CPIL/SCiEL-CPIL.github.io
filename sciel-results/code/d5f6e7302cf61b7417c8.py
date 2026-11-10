"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150

# EVOLVE-BLOCK-START

def _log_energy_and_grad(x):
    """
    Compute the logarithmic energy and its gradient for points on the unit sphere.
    x: (N, 3) array of points.
    Returns (energy, grad) where grad is the gradient in R^3 (not projected).
    """
    # x is (N, 3)
    # Compute pairwise distances
    # d_{ij} = |x_i - x_j|
    # We want sum_{i<j} log(1/d_{ij}) = -sum_{i<j} log(d_{ij})
    
    # For efficiency, compute all pairwise differences
    # diff_{ij} = x_i - x_j, shape (N, N, 3)
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]  # (N, N, 3)
    d2 = np.sum(diff * diff, axis=2)  # (N, N)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(d2, 1.0)
    
    d = np.sqrt(d2)  # (N, N)
    
    # Energy: sum_{i<j} log(1/d_{ij}) = -sum_{i<j} log(d_{ij})
    # We can compute the full matrix sum and divide by 2, but need to exclude diagonal
    log_d = np.log(d)  # (N, N)
    # Set diagonal to 0 so it doesn't contribute
    np.fill_diagonal(log_d, 0.0)
    energy = -0.5 * np.sum(log_d)
    
    # Gradient: d/dx_i of [-sum_{j != i} log(d_{ij})]
    # d/dx_i log(d_{ij}) = (x_i - x_j) / d_{ij}^2
    # So d/dx_i [-log(d_{ij})] = -(x_i - x_j) / d_{ij}^2
    # grad_i = -sum_{j != i} (x_i - x_j) / d_{ij}^2
    #         = sum_{j != i} (x_j - x_i) / d_{ij}^2
    
    # Compute (x_j - x_i) / d_{ij}^2 for each pair
    # inv_d2 = 1 / d_{ij}^2, shape (N, N)
    inv_d2 = 1.0 / d2  # (N, N)
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad_i = sum_j inv_d2_{ij} * (x_j - x_i)
    #        = sum_j inv_d2_{ij} * x_j - x_i * sum_j inv_d2_{ij}
    
    # sum_j inv_d2_{ij} * x_j: matrix-vector product
    # inv_d2 is (N, N), x is (N, 3)
    term1 = inv_d2 @ x  # (N, 3)
    
    # sum_j inv_d2_{ij}
    row_sums = np.sum(inv_d2, axis=1)  # (N,)
    
    grad = term1 - row_sums[:, np.newaxis] * x  # (N, 3)
    
    return energy, grad


def _project_to_sphere(x):
    """Project points back onto the unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms < 1e-15, 1.0, norms)
    return x / norms


def _fibonacci_spiral():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Start with Fibonacci spiral
    x0 = _fibonacci_spiral()
    
    # Flatten for scipy optimizer
    x0_flat = x0.flatten()
    
    def objective_and_grad(flat_x):
        x = flat_x.reshape(N, 3)
        # Project to sphere for evaluation
        x_proj = _project_to_sphere(x)
        energy, grad = _log_energy_and_grad(x_proj)
        # Project gradient to tangent space: grad_tan = grad - (grad . x) * x
        dot = np.sum(grad * x_proj, axis=1, keepdims=True)
        grad_tan = grad - dot * x_proj
        return energy, grad_tan.flatten()
    
    # Use L-BFGS-B for optimization
    # We'll do multiple restarts with different initial conditions if time permits
    
    best_x = x0.copy()
    best_energy = _log_energy_and_grad(best_x)[0]
    
    # First optimization from Fibonacci start
    try:
        res = minimize(
            objective_and_grad,
            x0_flat,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 1000, 'maxls': 50, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        x_opt = _project_to_sphere(res.x.reshape(N, 3))
        e_opt, _ = _log_energy_and_grad(x_opt)
        if e_opt < best_energy:
            best_energy = e_opt
            best_x = x_opt
    except Exception:
        pass
    
    # Check time
    if time.time() > deadline - 10.0:
        return best_x
    
    # Try additional restarts with perturbed starts
    rng = np.random.RandomState(42)
    
    num_restarts = 0
    while time.time() < deadline - 15.0 and num_restarts < 20:
        num_restarts += 1
        # Create a perturbed version of the best solution
        noise = rng.randn(N, 3) * 0.1
        x_pert = _project_to_sphere(best_x + noise)
        
        x_pert_flat = x_pert.flatten()
        
        try:
            res = minimize(
                objective_and_grad,
                x_pert_flat,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'maxls': 50, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            x_opt = _project_to_sphere(res.x.reshape(N, 3))
            e_opt, _ = _log_energy_and_grad(x_opt)
            if e_opt < best_energy:
                best_energy = e_opt
                best_x = x_opt
        except Exception:
            pass
    
    # Final check: ensure points are on sphere and valid
    best_x = _project_to_sphere(best_x)
    
    # Verify no points are too close
    # (The optimizer should handle this, but just in case)
    
    return best_x

# EVOLVE-BLOCK-END
