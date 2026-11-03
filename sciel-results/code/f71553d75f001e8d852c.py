"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _energy(X):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(X)
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    idx = np.triu_indices(n, k=1)
    d = dist[idx]
    d = np.maximum(d, 1e-15)
    return -np.sum(np.log(d))


def _energy_and_grad(X):
    """Compute logarithmic energy and gradient on sphere (vectorized)."""
    n = len(X)
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    idx = np.triu_indices(n, k=1)
    d = dist[idx]
    d = np.maximum(d, 1e-15)
    E = -np.sum(np.log(d))
    
    # Gradient: dE/dx_i = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # 1/d^2 for each pair
    inv_d2 = np.zeros_like(dist_sq)
    np.fill_diagonal(inv_d2, 0.0)
    inv_d2 = 1.0 / np.maximum(dist_sq, 1e-30)
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad[i] = -sum_j (X[i] - X[j]) / d_ij^2
    # = -X[i] * sum_j(1/d_ij^2) + sum_j X[j] / d_ij^2
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    grad = -X * sum_inv_d2[:, np.newaxis]
    for j in range(n):
        grad += inv_d2[:, j] * X[j]
    
    return E, grad


def _project_to_sphere(X):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms


def _fibonacci_sphere():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(seed=42):
    """Random points on unit sphere."""
    rng = np.random.RandomState(seed)
    X = rng.randn(N, 3)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X


def _riemannian_step(X, grad, alpha):
    """Take a Riemannian step with backtracking."""
    # Project gradient to tangent space
    proj_grad = grad - np.sum(grad * X, axis=1, keepdims=True) * X
    direction = -proj_grad
    
    dir_norms = np.linalg.norm(direction, axis=1, keepdims=True)
    dir_norms = np.maximum(dir_norms, 1e-15)
    direction = direction / dir_norms
    
    best_X = X.copy()
    best_E = _energy(X)
    
    a = alpha
    for _ in range(50):
        step = a * direction
        step_norms = np.linalg.norm(step, axis=1, keepdims=True)
        step_norms = np.maximum(step_norms, 1e-15)
        step_unit = step / step_norms
        
        cos_t = np.cos(step_norms)
        sin_t = np.sin(step_norms)
        X_new = X * cos_t + step_unit * sin_t
        
        E_new = _energy(X_new)
        if E_new < best_E:
            best_X = X_new.copy()
            best_E = E_new
        else:
            a *= 0.5
            if a < 1e-14:
                break
    
    return best_X, best_E


def _gradient_descent(X, max_iter=5000, tol=1e-12, initial_alpha=0.1):
    """Riemannian gradient descent on the sphere."""
    best_X = X.copy()
    best_E = _energy(X)
    alpha = initial_alpha
    
    for it in range(max_iter):
        E, grad = _energy_and_grad(best_X)
        
        proj_grad = grad - np.sum(grad * best_X, axis=1, keepdims=True) * best_X
        grad_norm = np.linalg.norm(proj_grad)
        
        if grad_norm < tol:
            break
        
        X_new, E_new = _riemannian_step(best_X, grad, alpha)
        
        if E_new < best_E - 1e-15:
            best_X = X_new
            best_E = E_new
            alpha = min(alpha * 1.1, 0.5)
        else:
            alpha *= 0.5
    
    return best_X, best_E


def _polish(X, max_iter=5000, tol=1e-14):
    """High-precision polish pass."""
    best_X = X.copy()
    best_E = _energy(X)
    alpha = 0.01
    
    for it in range(max_iter):
        E, grad = _energy_and_grad(best_X)
        
        proj_grad = grad - np.sum(grad * best_X, axis=1, keepdims=True) * best_X
        grad_norm = np.linalg.norm(proj_grad)
        
        if grad_norm < tol:
            break
        
        X_new, E_new = _riemannian_step(best_X, grad, alpha)
        
        if E_new < best_E - 1e-16:
            best_X = X_new
            best_E = E_new
        else:
            alpha *= 0.5
    
    return best_X, best_E


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Start with Fibonacci sphere
    X = _fibonacci_sphere()
    best_X = X.copy()
    best_E = _energy(X)
    
    # Gradient descent from Fibonacci start
    X_fd, E_fd = _gradient_descent(best_X, max_iter=3000, tol=1e-12)
    if E_fd < best_E:
        best_X = X_fd
        best_E = E_fd
    
    # Try random starts if time permits
    random_seeds = [1, 2, 3, 7, 42, 123, 999, 2024, 555, 888]
    for seed in random_seeds:
        if time.time() > deadline - 15.0:
            break
        Xr = _random_sphere(seed)
        Xr_gd, E_r = _gradient_descent(Xr, max_iter=2000, tol=1e-12)
        if E_r < best_E:
            best_X = Xr_gd
            best_E = E_r
    
    # Final polish
    if time.time() < deadline - 3.0:
        best_X, best_E = _polish(best_X, max_iter=10000, tol=1e-14)
    
    # Ensure points are on unit sphere
    best_X = _project_to_sphere(best_X)
    
    return best_X
# EVOLVE-BLOCK-END
