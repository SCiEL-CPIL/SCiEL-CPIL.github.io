"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 150
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0  # seconds before wall-clock kill


def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=42):
    """Generate random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    phi = np.arccos(rng.uniform(-1, 1, n))
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])


def _energy_and_grad(X):
    """
    Compute logarithmic energy and its gradient on the sphere.
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    Gradient for point i: grad_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    (this is the Euclidean gradient; for Riemannian we project to tangent space)
    """
    n = X.shape[0]
    # Compute pairwise distances
    # d_ij = |x_i - x_j|
    # Use broadcasting
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    np.fill_diagonal(dist_sq, 1.0)  # avoid log(0)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of log(1/dist_ij) = -sum_{i<j} log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dist[iu])
    energy = -np.sum(log_dists)
    
    # Gradient: for each i, grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # diff[i,j] = x_i - x_j
    # grad_i = sum_j (x_i - x_j) / dist_sq[i,j]
    # = sum_j diff[i,j] / dist_sq[i,j]
    inv_dist_sq = 1.0 / dist_sq  # (n, n)
    inv_dist_sq[np.diag_indices(n)] = 0.0
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)  # (n, 3)
    
    return energy, grad


def _riemannian_step(X, grad, step_size, momentum=None, beta=0.9):
    """
    Perform a Riemannian gradient descent step using the exponential map.
    The tangent vector is the projection of the Euclidean gradient onto the tangent space.
    For the sphere, tangent at x is {v : v . x = 0}.
    exp_x(v) = x*cos(|v|) + (x_hat)*sin(|v|) where x_hat = v/|v|
    But since we project grad to tangent: v = grad - (grad . x) x
    """
    n = X.shape[0]
    # Project gradient to tangent space
    # grad_proj_i = grad_i - (grad_i . x_i) x_i
    dot = np.sum(grad * X, axis=1)  # (n,)
    tangent = grad - dot[:, np.newaxis] * X  # (n, 3)
    
    # Apply momentum if provided
    if momentum is not None:
        v = -step_size * (tangent + beta * momentum)
        new_momentum = beta * momentum + (1 - beta) * tangent
    else:
        v = -step_size * tangent  # negative gradient direction
        new_momentum = None
    
    v_norm = np.linalg.norm(v, axis=1)
    v_norm_safe = np.where(v_norm > 1e-15, v_norm, 1.0)
    
    # Exponential map: exp_x(v) = x * cos(|v|) + (v/|v|) * sin(|v|)
    cos_t = np.cos(v_norm_safe)
    sin_t = np.sin(v_norm_safe)
    X_new = X * cos_t[:, np.newaxis] + (v / v_norm_safe[:, np.newaxis]) * sin_t[:, np.newaxis]
    
    # Handle cases where v_norm was zero
    zero_mask = v_norm < 1e-15
    X_new[zero_mask] = X[zero_mask]
    
    # Re-normalize to ensure unit sphere
    norms = np.linalg.norm(X_new, axis=1)
    X_new = X_new / norms[:, np.newaxis]
    
    return X_new, new_momentum


def _line_search(X, grad, energy0, max_iter=20, initial_step=0.1, momentum=None, beta=0.9):
    """
    Backtracking line search for Riemannian gradient descent.
    Takes energy0 as input to avoid recomputation.
    """
    step = initial_step
    for _ in range(max_iter):
        X_new, new_momentum = _riemannian_step(X, grad, step, momentum=momentum, beta=beta)
        energy_new, _ = _energy_and_grad(X_new)
        if energy_new < energy0:
            return X_new, step, new_momentum
        step *= 0.5
    return X, initial_step, momentum


def construct():
    start_time = time.time()
    
    # Start with Fibonacci sphere
    X = _fibonacci_sphere(N)
    
    # Compute initial energy
    best_energy, _ = _energy_and_grad(X)
    best_X = X.copy()
    
    # Riemannian gradient descent with momentum and adaptive step size
    step_size = 0.05
    min_step = 1e-12
    max_iterations = 10000
    momentum = np.zeros_like(X)
    beta = 0.9
    
    for it in range(max_iterations):
        # Check time
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Compute energy and gradient
        energy, grad = _energy_and_grad(X)
        
        # Update best
        if energy < best_energy:
            best_energy = energy
            best_X = X.copy()
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
        
        # Line search with Riemannian step
        X_new, step_size, momentum = _line_search(X, grad, energy, max_iter=15, 
                                                   initial_step=step_size, 
                                                   momentum=momentum, beta=beta)
        
        if np.allclose(X, X_new, atol=1e-15):
            break
        
        X = X_new
        step_size = max(step_size, min_step)
    
    # Final high-precision polish with smaller steps
    elapsed = time.time() - start_time
    remaining = TIME_LIMIT - SAFETY_MARGIN - elapsed
    if remaining > 2.0:
        X = best_X.copy()
        momentum = np.zeros_like(X)
        for _ in range(500):
            elapsed = time.time() - start_time
            if elapsed > TIME_LIMIT - SAFETY_MARGIN:
                break
            energy, grad = _energy_and_grad(X)
            if energy < best_energy:
                best_energy = energy
                best_X = X.copy()
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-16:
                break
            # Use adaptive step based on gradient magnitude
            step = min(0.01, 0.1 / (grad_norm + 1e-10))
            X_new, new_momentum = _riemannian_step(X, grad, step, momentum=momentum, beta=0.9)
            energy_new, _ = _energy_and_grad(X_new)
            if energy_new < best_energy:
                best_energy = energy_new
                best_X = X_new.copy()
                X = X_new
                momentum = new_momentum
            else:
                # Try smaller step
                step = 0.001
                X_new, new_momentum = _riemannian_step(X, grad, step, momentum=momentum, beta=0.9)
                energy_new, _ = _energy_and_grad(X_new)
                if energy_new < best_energy:
                    best_energy = energy_new
                    best_X = X_new.copy()
                    X = X_new
                    momentum = new_momentum
    
    return best_X
