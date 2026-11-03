"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _energy_and_grad(X):
    """Compute logarithmic energy and its gradient for points X on the unit sphere."""
    N = X.shape[0]
    # Compute pairwise distances
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Avoid division by zero for i==j
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    log_dist = np.log(dist[mask])
    energy = -np.sum(log_dist)
    
    # Gradient: d/dX_i [ -log|X_i - X_j| ] = (X_i - X_j) / |X_i - X_j|^2
    # For each point i, grad_i = sum_{j != i} (X_i - X_j) / |X_i - X_j|^2
    dist2 = dist * dist
    np.fill_diagonal(dist2, 1.0)
    inv_dist2 = 1.0 / dist2
    np.fill_diagonal(inv_dist2, 0.0)
    
    # grad[i] = sum_j (X_i - X_j) / |X_i - X_j|^2
    # = sum_j X_i / dist2_ij - sum_j X_j / dist2_ij
    # = X_i * sum_j (1/dist2_ij) - sum_j X_j / dist2_ij
    sum_inv = inv_dist2.sum(axis=1)  # sum_j 1/dist2_ij for each i
    weighted_sum = (inv_dist2[:, :, np.newaxis] * X[np.newaxis, :, :]).sum(axis=1)
    
    grad = X * sum_inv[:, np.newaxis] - weighted_sum
    
    # Project gradient onto tangent space (remove radial component)
    # tangent_grad = grad - (grad . X) * X
    dot = (grad * X).sum(axis=1, keepdims=True)
    grad = grad - dot * X
    
    return energy, grad


def _project_to_sphere(X):
    """Project points back onto the unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return X / norms


def _fibonacci_spiral():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _gradient_descent_with_momentum(X, lr=0.01, momentum=0.9, steps=500, deadline=None):
    """Gradient descent with momentum on the sphere using Riemannian exponential map."""
    X = X.copy()
    velocity = np.zeros_like(X)
    best_energy = float('inf')
    best_X = X.copy()
    
    for step in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        
        energy, grad = _energy_and_grad(X)
        if energy < best_energy:
            best_energy = energy
            best_X = X.copy()
        
        # Update velocity
        velocity = momentum * velocity - lr * grad
        
        # Riemannian exponential map: move along tangent direction
        # v_tangent is already tangent (from _energy_and_grad)
        # X_new = cos(|v|) * X + sin(|v|)/|v| * v
        v_norms = np.linalg.norm(velocity, axis=1, keepdims=True)
        v_norms_safe = np.where(v_norms < 1e-15, 1.0, v_norms)
        
        cos_term = np.cos(v_norms_safe)
        sin_term = np.sin(v_norms_safe) / v_norms_safe
        
        X = cos_term * X + sin_term * velocity
        
        # Ensure points stay on sphere
        X = _project_to_sphere(X)
        
        # Adaptive learning rate: reduce if energy increases
        if step > 10 and energy > best_energy * 1.01:
            lr *= 0.5
    
    return best_X, best_energy


def _lbfgs_polish(X, deadline=None):
    """High-precision L-BFGS-B polish using parameterization X -> normalize(X + epsilon*tangent)."""
    N = X.shape[0]
    
    def objective_flat(flat_X):
        X = flat_X.reshape(N, 3)
        X = _project_to_sphere(X)
        energy, _ = _energy_and_grad(X)
        return energy
    
    def gradient_flat(flat_X):
        X = flat_X.reshape(N, 3)
        X = _project_to_sphere(X)
        _, grad = _energy_and_grad(X)
        # Project gradient to ambient space for L-BFGS-B
        return grad.flatten()
    
    x0 = X.flatten()
    
    result = minimize(
        objective_flat,
        x0,
        jac=gradient_flat,
        method='L-BFGS-B',
        options={
            'maxiter': 200,
            'ftol': 1e-12,
            'gtol': 1e-10,
            'maxfun': 500
        }
    )
    
    X_opt = _project_to_sphere(result.x.reshape(N, 3))
    energy, _ = _energy_and_grad(X_opt)
    
    return X_opt, energy


def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 110.0  # 10 second safety margin
    
    # Start with Fibonacci spiral
    X = _fibonacci_spiral()
    
    # Phase 1: Gradient descent with momentum
    X, energy = _gradient_descent_with_momentum(X, lr=0.01, momentum=0.9, steps=300, deadline=deadline)
    
    if time.time() > deadline - 30.0:
        # Not enough time for polish, return current best
        return X
    
    # Phase 2: L-BFGS-B polish
    X, energy = _lbfgs_polish(X, deadline=deadline)
    
    # Final validation: ensure all points are valid
    norms = np.linalg.norm(X, axis=1)
    if np.any(norms < 1e-9) or not np.all(np.isfinite(X)):
        # Fallback to Fibonacci spiral if something went wrong
        return _fibonacci_spiral()
    
    return X
# EVOLVE-BLOCK-END
