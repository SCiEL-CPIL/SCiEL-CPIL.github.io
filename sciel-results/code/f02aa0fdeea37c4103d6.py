"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _log_energy(X):
    """Compute the logarithmic energy sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(X)
    # Compute pairwise distances
    # Use broadcasting for efficiency
    # X shape (n, 3)
    # dist[i,j] = |X[i] - X[j]|
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    # Set diagonal to 1 to avoid log(0)
    np.fill_diagonal(dist, 1.0)
    # Compute log(1/d) = -log(d) for all pairs, then sum upper triangle
    log_inv_dist = -np.log(dist)
    # Sum only i < j
    iu = np.triu_indices(n, k=1)
    energy = math.fsum(log_inv_dist[iu])
    return energy


def _log_energy_and_grad(X):
    """Compute logarithmic energy and its gradient."""
    n = len(X)
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} -log(dist_ij)
    iu = np.triu_indices(n, k=1)
    log_inv_dist = -np.log(dist[iu])
    energy = math.fsum(log_inv_dist)
    
    # Gradient: d/dX_i sum_{j!=i} -log(|X_i - X_j|)
    # = sum_{j!=i} (X_i - X_j) / |X_i - X_j|^2
    # For each i, grad_i = sum_{j!=i} (X_i - X_j) / dist_ij^2
    # Note: dist^2 = dist_sq
    # grad shape (n, 3)
    # We can compute this as: for each pair (i,j), contribution to grad_i is (X_i-X_j)/dist^2, to grad_j is (X_j-X_i)/dist^2 = -(X_i-X_j)/dist^2
    # So grad = sum over j!=i of (X_i - X_j)/dist_ij^2
    
    # Efficient computation:
    # Let A[i,j] = 1/dist_ij^2 for i!=j, 0 for i==j
    # Then grad_i = sum_j A[i,j] * (X_i - X_j) = X_i * sum_j A[i,j] - sum_j A[i,j] * X_j
    # Let S_i = sum_j A[i,j] (row sum)
    # Let B = A @ X (matrix-vector product), so B_i = sum_j A[i,j] X_j
    # grad_i = S_i * X_i - B_i
    
    A = 1.0 / dist_sq
    np.fill_diagonal(A, 0.0)
    S = A.sum(axis=1)
    B = A @ X
    grad = S[:, np.newaxis] * X - B
    
    return energy, grad


def _project_to_sphere(X):
    """Project points to unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return X / norms


def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _optimize_from_start(X0, max_iter=2000, tol=1e-10):
    """Optimize point configuration starting from X0 using gradient descent with projection."""
    X = X0.copy()
    n = len(X)
    
    # Adaptive learning rate
    lr = 0.01
    best_energy = _log_energy(X)
    best_X = X.copy()
    
    for it in range(max_iter):
        energy, grad = _log_energy_and_grad(X)
        
        # Gradient step
        X_new = X - lr * grad
        
        # Project to sphere
        X_new = _project_to_sphere(X_new)
        
        # Check if energy improved
        new_energy = _log_energy(X_new)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_X = X_new.copy()
        
        # Adaptive learning rate
        if new_energy < energy:
            lr = min(lr * 1.05, 0.1)
        else:
            lr = max(lr * 0.8, 1e-6)
        
        X = X_new
        
        # Convergence check
        if np.linalg.norm(grad) < tol:
            break
    
    return best_X, best_energy


def construct():
    """Construct 100 points on the unit sphere to minimize logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Start from Fibonacci spiral
    X0 = _fibonacci_sphere(N)
    
    # Multiple random restarts
    best_X = X0.copy()
    best_energy = _log_energy(X0)
    
    # First optimize from Fibonacci start
    X_opt, E_opt = _optimize_from_start(X0, max_iter=3000, tol=1e-12)
    if E_opt < best_energy:
        best_energy = E_opt
        best_X = X_opt
    
    # Check time
    if time.time() > deadline - 30:
        return _project_to_sphere(best_X)
    
    # Random restarts
    rng = np.random.default_rng(42)
    for restart in range(5):
        if time.time() > deadline - 20:
            break
        
        # Random start with some structure
        X_rand = rng.standard_normal((N, 3))
        X_rand = _project_to_sphere(X_rand)
        
        X_opt, E_opt = _optimize_from_start(X_rand, max_iter=2000, tol=1e-10)
        if E_opt < best_energy:
            best_energy = E_opt
            best_X = X_opt
        
        # Also try perturbing the best solution
        if time.time() > deadline - 15:
            break
        X_perturbed = best_X + rng.standard_normal((N, 3)) * 0.1
        X_perturbed = _project_to_sphere(X_perturbed)
        X_opt, E_opt = _optimize_from_start(X_perturbed, max_iter=1500, tol=1e-10)
        if E_opt < best_energy:
            best_energy = E_opt
            best_X = X_opt
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 10:
        X_final, E_final = _optimize_from_start(best_X, max_iter=5000, tol=1e-14)
        if E_final < best_energy:
            best_X = X_final
    
    # Ensure valid output
    best_X = _project_to_sphere(best_X)
    # Check for any zero or non-finite values
    if not np.all(np.isfinite(best_X)) or np.any(np.linalg.norm(best_X, axis=1) < 1e-15):
        return _fibonacci_sphere(N)
    
    return best_X
# EVOLVE-BLOCK-END
