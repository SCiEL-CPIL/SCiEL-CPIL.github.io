"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _fibonacci_spiral(n, seed_offset=0.0):
    """Generate n points on the unit sphere using a Fibonacci spiral with optional offset."""
    i = np.arange(n) + 0.5 + seed_offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _log_energy_and_grad(X):
    """
    Compute the logarithmic energy E = sum_{i<j} log(1/|x_i - x_j|)
    and its gradient with respect to X (N x 3).
    
    Energy = -sum_{i<j} log(d_ij)
    Gradient w.r.t. x_i: sum_{j != i} (x_i - x_j) / d_ij^2
    """
    # Compute all pairwise distances
    # X is (N, 3)
    # diff[i,j] = x_i - x_j
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (N, N, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (N, N)
    
    # Avoid division by zero and log(0)
    # Set diagonal to 1 to avoid log(1)=0 and division issues, but we mask it out
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Mask for i < j
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    
    # Energy: sum_{i<j} -log(d_ij)
    d_pairs = dist[mask]
    energy = -np.sum(np.log(d_pairs))
    
    # Gradient: for each i, grad_i = sum_{j != i} (x_i - x_j) / d_ij^2
    # We can compute this efficiently:
    # grad[i] = sum_j (x_i - x_j) / dist_sq[i,j]
    # Note: for j=i, term is 0/1 = 0, so we don't need to exclude diagonal explicitly if we handle it
    # But dist_sq diagonal is 1.0, and diff[i,i] = 0, so it's fine.
    
    # grad = (diff / dist_sq[:, :, np.newaxis]).sum(axis=1)
    # But we need to be careful: the term (x_i - x_j)/d^2 is symmetric in sign.
    # Let's compute:
    # inv_dist_sq = 1.0 / dist_sq
    # inv_dist_sq[np.diag_indices(N)] = 0.0  # not needed since diff[i,i]=0
    # grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    inv_dist_sq = 1.0 / dist_sq
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(X):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms


def construct():
    t_start = time.time()
    deadline = t_start + 145.0  # Safety margin
    
    best_X = _fibonacci_spiral(N)
    best_energy, _ = _log_energy_and_grad(best_X)
    
    # Multi-start Riemannian gradient descent
    n_starts = 20
    
    for start_idx in range(n_starts):
        if time.time() > deadline:
            break
        
        # Generate initial configuration
        if start_idx == 0:
            X = _fibonacci_spiral(N)
        else:
            # Perturbed Fibonacci spiral
            offset = np.random.uniform(-0.5, 0.5)
            X = _fibonacci_spiral(N, seed_offset=offset)
            # Add small random perturbation
            X += 0.01 * np.random.randn(N, 3)
            X = _project_to_sphere(X)
        
        # Use scipy minimize with BFGS
        # We'll use a loop of gradient steps with adaptive step size
        
        X_curr = X.copy()
        energy_curr, grad_curr = _log_energy_and_grad(X_curr)
        
        # Riemannian gradient descent with backtracking line search
        step_size = 0.01
        max_iter = 500
        tol = 1e-10
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Project gradient to tangent space
            # Tangent space: v such that v . x_i = 0 for all i
            # grad_proj = grad - (grad . x_i) x_i
            dot_products = np.sum(grad_curr * X_curr, axis=1)
            grad_tangent = grad_curr - dot_products[:, np.newaxis] * X_curr
            
            # Take a step
            X_new = X_curr - step_size * grad_tangent
            X_new = _project_to_sphere(X_new)
            
            energy_new, grad_new = _log_energy_and_grad(X_new)
            
            # Check if energy decreased
            if energy_new < energy_curr:
                X_curr = X_new
                energy_curr = energy_new
                grad_curr = grad_new
                step_size = min(step_size * 1.5, 0.1)
            else:
                step_size *= 0.5
                if step_size < 1e-8:
                    break
            
            # Check convergence
            if np.linalg.norm(grad_tangent) < tol:
                break
        
        # Update best
        if energy_curr < best_energy:
            best_energy = energy_curr
            best_X = X_curr.copy()
    
    # Final polish: use scipy's minimize with L-BFGS-B on the sphere-constrained problem
    # We can use a parameterization that automatically satisfies the constraint,
    # or use penalty methods. Here we'll do a few more steps of gradient descent.
    
    if time.time() < deadline - 5.0:
        X_curr = best_X.copy()
        energy_curr, grad_curr = _log_energy_and_grad(X_curr)
        
        step_size = 0.001
        for it in range(1000):
            if time.time() > deadline - 1.0:
                break
            
            dot_products = np.sum(grad_curr * X_curr, axis=1)
            grad_tangent = grad_curr - dot_products[:, np.newaxis] * X_curr
            
            X_new = X_curr - step_size * grad_tangent
            X_new = _project_to_sphere(X_new)
            
            energy_new, grad_new = _log_energy_and_grad(X_new)
            
            if energy_new < energy_curr:
                X_curr = X_new
                energy_curr = energy_new
                grad_curr = grad_new
            else:
                step_size *= 0.5
                if step_size < 1e-12:
                    break
            
            if np.linalg.norm(grad_tangent) < 1e-12:
                break
        
        if energy_curr < best_energy:
            best_energy = energy_curr
            best_X = X_curr
    
    # Ensure all points are on the unit sphere and valid
    best_X = _project_to_sphere(best_X)
    
    # Check for any issues
    norms = np.linalg.norm(best_X, axis=1)
    if not np.all(np.isfinite(best_X)) or np.any(norms < 1e-9):
        # Fallback to Fibonacci spiral
        best_X = _fibonacci_spiral(N)
    
    return best_X
# EVOLVE-BLOCK-END
