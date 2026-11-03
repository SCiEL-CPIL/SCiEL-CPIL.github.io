"""Optimize 100 points on sphere to minimize logarithmic energy."""
import numpy as np
import time
import math

N = 100
DEADLINE = 140.0  # safety margin before 150s kill

def _energy_and_grad(X):
    """Compute log-energy and gradient for points on sphere.
    X: (N, 3) array of unit vectors.
    Returns (energy, grad) where grad is the Riemannian gradient (tangent).
    """
    # Pairwise distances
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (N, N, 3)
    dist2 = np.sum(diff * diff, axis=2)  # (N, N)
    dist = np.sqrt(np.maximum(dist2, 1e-30))
    
    # Avoid self-pairs
    np.fill_diagonal(dist, np.inf)
    np.fill_diagonal(dist2, np.inf)
    
    # Energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
    # Using full matrix, divide by 2 for i<j
    log_dist = np.log(dist)
    energy = -0.5 * np.sum(log_dist)
    
    # Gradient of energy w.r.t. X_i:
    # dE/dX_i = sum_{j != i} (X_i - X_j) / dist_ij^2 * (-1) ... let me derive
    # E = -sum_{i<j} log|xi - xj|
    # dE/dxi = -sum_{j!=i, j>i} (xi-xj)/|xi-xj|^2 - sum_{j<i} (xi-xj)/|xi-xj|^2
    #        = -sum_{j!=i} (xi-xj)/|xi-xj|^2
    # But we want the Riemannian gradient (project onto tangent space)
    # Euclidean grad: g_i = -sum_{j!=i} (xi - xj) / dist_ij^2
    # Wait, let me be more careful.
    # E = -1/2 * sum_{i,j, i!=j} log|xi-xj|
    # dE/dxi = -1/2 * sum_{j!=i} (xi-xj)/|xi-xj|^2 * 2 = -sum_{j!=i} (xi-xj)/|xi-xj|^2
    # Actually: d/dxi log|xi-xj| = (xi-xj)/|xi-xj|^2
    # So dE/dxi = -sum_{j!=i} (xi-xj)/|xi-xj|^2
    
    # Compute pairwise: for each i, sum over j!=i of (xi-xj)/dist^2
    # (xi-xj)/dist^2 = diff[i,j] / dist2[i,j]
    inv_dist2 = np.zeros_like(dist2)
    mask = ~np.isinf(dist2)
    inv_dist2[mask] = 1.0 / dist2[mask]
    
    # grad_i = -sum_j diff[i,j] * inv_dist2[i,j]
    # diff[i,j] = xi - xj
    grad = -np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)  # (N, 3)
    
    # Project to tangent space: grad_tangent = grad - (grad . xi) * xi
    dot = np.sum(grad * X, axis=1, keepdims=True)
    grad_tangent = grad - dot * X
    
    return energy, grad_tangent


def _exponential_map(X, V):
    """Exponential map on sphere: move from X along tangent vector V.
    X: (N, 3) unit vectors, V: (N, 3) tangent vectors.
    Returns new unit vectors.
    """
    theta = np.linalg.norm(V, axis=1, keepdims=True)
    # Avoid division by zero
    theta_safe = np.where(theta < 1e-15, 1.0, theta)
    sin_theta = np.sin(theta_safe)
    cos_theta = np.cos(theta_safe)
    # exp_X(V) = cos(theta)*X + sin(theta)/theta * V
    factor = sin_theta / theta_safe
    factor = np.where(theta < 1e-15, 1.0, factor)
    result = cos_theta * X + factor * V
    # Renormalize
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return result / norms


def _retract(X, V):
    """Retraction: normalize X + V (first-order approximation of exp map)."""
    result = X + V
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return result / norms


def construct():
    t0 = time.time()
    
    # Initial: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure unit norm
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    energy, grad = _energy_and_grad(X)
    best_X = X.copy()
    best_energy = energy
    
    # Riemannian gradient descent with momentum
    momentum = np.zeros_like(X)
    beta1 = 0.9
    beta2 = 0.999
    m = np.zeros_like(X)
    v = np.zeros_like(X)
    t = 0
    
    # Phase 1: Adaptive learning rate gradient descent
    lr = 0.01
    patience = 0
    best_energy_phase1 = energy
    
    while time.time() - t0 < DEADLINE - 30.0:  # leave 30s for Newton polish
        t += 1
        t_hat = t / (1 - beta1**t)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad * grad)
        v_hat = v / (1 - beta2**t)
        
        # Adaptive step size
        step = lr / (np.sqrt(v_hat + 1e-8) + 1e-8)
        update = m / t_hat * step
        
        # Try the step
        X_new = _exponential_map(X, -update)
        E_new, g_new = _energy_and_grad(X_new)
        
        if E_new < energy - 1e-12:
            X = X_new
            energy = E_new
            grad = g_new
            momentum = update
            patience = 0
            if energy < best_energy:
                best_energy = energy
                best_X = X.copy()
            if energy < best_energy_phase1 - 1e-6:
                best_energy_phase1 = energy
                lr = lr * 1.05  # slightly increase
        else:
            patience += 1
            lr = lr * 0.9
            if patience > 50:
                # Try a larger restart step
                lr = 0.01
                patience = 0
        
        # Check convergence
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10 and patience > 20:
            break
    
    X = best_X.copy()
    energy, grad = _energy_and_grad(X)
    
    # Phase 2: Newton polish
    # For Newton on sphere, we need Hessian of the energy.
    # Hessian of E = -sum_{i<j} log|xi-xj|
    # This is complex. Instead, use a few steps of steepest descent with very small lr,
    # or use the fact that near optimum, the problem is approximately quadratic.
    
    # Alternative: use scipy's minimize with constraints if available
    # But let's do a simple high-precision polish with line search
    
    t0_polish = time.time()
    polish_lr = 1e-4
    for it in range(200):
        if time.time() - t0_polish > 25.0:
            break
        E, g = _energy_and_grad(X)
        if np.linalg.norm(g) < 1e-12:
            break
        
        # Backtracking line search
        alpha = polish_lr
        for ls in range(30):
            X_try = _exponential_map(X, -alpha * g)
            E_try, _ = _energy_and_grad(X_try)
            if E_try < E - 1e-14 * alpha * np.sum(g * g):
                break
            alpha *= 0.5
        
        X = X_try
        if E_try < energy:
            energy = E_try
            best_X = X.copy()
        else:
            break
    
    X = best_X
    energy, _ = _energy_and_grad(X)
    
    # Final: ensure all points are distinct and on sphere
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    # Validate: no points closer than 1e-9
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    if min_dist < 1e-9:
        # Perturb slightly
        rng = np.random.RandomState(42)
        for _ in range(100):
            X += rng.normal(0, 1e-6, X.shape)
            X = X / np.linalg.norm(X, axis=1, keepdims=True)
            diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff * diff, axis=2))
            np.fill_diagonal(dist, np.inf)
            if np.min(dist) > 1e-9:
                break
    
    return X
