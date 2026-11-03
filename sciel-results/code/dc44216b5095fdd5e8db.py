"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 100

# EVOLVE-BLOCK-START
def _compute_energy(X):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)"""
    n = X.shape[0]
    # Pairwise distances
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    # Avoid log(0)
    np.fill_diagonal(dists, 1.0)
    # Only upper triangle
    iu = np.triu_indices(n, k=1)
    d = dists[iu]
    # Use math.fsum for precision as specified
    terms = [-math.log(d[i]) for i in range(len(d))]
    return math.fsum(terms)


def _compute_energy_gradient(X):
    """Compute gradient of the logarithmic energy with respect to X.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = X.shape[0]
    grad = np.zeros_like(X)
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dists_sq, 1.0)
    inv_dists_sq = 1.0 / dists_sq
    # grad[i] = sum_j (x_i - x_j) / |x_i - x_j|^2
    grad = np.sum(diff * inv_dists_sq[:, :, np.newaxis], axis=1)
    return grad


def _normalize(X):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms


def _project_to_sphere(X):
    """Project points onto unit sphere."""
    return _normalize(X)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Start with golden angle spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    X = _normalize(X)
    
    best_X = X.copy()
    best_E = _compute_energy(X)
    
    # Multi-stage optimization:
    # Stage 1: Coarse gradient descent with adaptive step size
    # Stage 2: L-BFGS-B with sphere constraints
    # Stage 3: Fine polishing with tighter tolerances
    
    # Stage 1: Gradient descent
    lr = 0.01
    patience = 0
    max_patience = 200
    
    while time.time() < deadline - 30.0 and patience < max_patience:
        E = _compute_energy(X)
        if E < best_E:
            best_E = E
            best_X = X.copy()
            patience = 0
        else:
            patience += 1
        
        if patience >= max_patience:
            break
        
        grad = _compute_energy_gradient(X)
        X = X - lr * grad
        X = _normalize(X)
        
        # Adaptive learning rate
        if patience < 50:
            lr *= 0.995
        else:
            lr *= 0.99
    
    # Stage 2: Use scipy L-BFGS-B with bounds to keep points on sphere
    # We parameterize using spherical coordinates or just use gradient descent
    # with stronger normalization
    
    # Try multiple restarts with different initializations
    remaining_time = deadline - time.time()
    
    if remaining_time > 20.0:
        # Try L-BFGS-B style optimization using scipy
        from scipy.optimize import minimize
        
        def obj_and_grad(x_flat):
            X = x_flat.reshape(N, 3)
            X = _normalize(X)
            E = _compute_energy(X)
            grad = _compute_energy_gradient(X)
            # Project gradient to tangent space
            X_norm = X / np.linalg.norm(X, axis=1, keepdims=True)
            grad_tangent = grad - np.sum(grad * X_norm, axis=1, keepdims=True) * X_norm
            return E, grad_tangent.flatten()
        
        x_flat = best_X.flatten()
        
        # Run L-BFGS-B with time limit
        result = minimize(
            lambda x: obj_and_grad(x)[0],
            x_flat,
            jac=lambda x: obj_and_grad(x)[1],
            method='L-BFGS-B',
            options={'maxiter': 500, 'maxfun': 1000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        X_new = _normalize(result.x.reshape(N, 3))
        E_new = _compute_energy(X_new)
        if E_new < best_E:
            best_E = E_new
            best_X = X_new.copy()
        
        remaining_time = deadline - time.time()
    
    if remaining_time > 10.0:
        # Stage 3: Fine polishing with small steps
        lr = 0.001
        for _ in range(200):
            if time.time() > deadline - 5.0:
                break
            grad = _compute_energy_gradient(best_X)
            step = lr * grad
            # Adaptive step
            X_trial = _normalize(best_X - step)
            E_trial = _compute_energy(X_trial)
            if E_trial < best_E:
                best_E = E_trial
                best_X = X_trial
                lr = min(lr * 1.05, 0.01)
            else:
                lr *= 0.8
    
    # Final normalization and validation
    best_X = _normalize(best_X)
    
    # Ensure no points are too close
    diff = best_X[:, np.newaxis, :] - best_X[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    min_dist = np.min(dists)
    
    if min_dist < 1e-9:
        # Fallback to golden spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        best_X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        best_X = _normalize(best_X)
    
    return best_X
# EVOLVE-BLOCK-END
