"""Optimize 150 points on the unit sphere to minimize logarithmic energy."""
import numpy as np
from scipy.optimize import minimize
import time

N = 150
TIME_LIMIT = 110.0  # safety margin before 120s kill

def _energy_and_grad(X):
    """Compute logarithmic energy and its gradient.
    
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_i: sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    
    # Compute pairwise differences
    # diff[i,j] = x_i - x_j
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff ** 2, axis=-1))
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    
    # Energy: -sum_{i<j} log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    E = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each i, grad_i = sum_{j != i} (x_i - x_j) / dist_ij^2
    # diff[i,j] / dist[i,j]^2
    inv_dist_sq = 1.0 / (dist ** 2)
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return E, grad

def _project(X):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms

def _fibonacci_init():
    """Golden-angle Fibonacci spiral initialization."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_init(seed=42):
    """Random initialization on sphere."""
    rng = np.random.RandomState(seed)
    X = rng.randn(N, 3)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _optimize_from(X0, time_start, label=""):
    """Run L-BFGS-B optimization from X0, with time guard."""
    best_X = X0.copy()
    best_E, _ = _energy_and_grad(X0)
    
    # Flatten for scipy
    x_flat = X0.flatten()
    
    def obj_and_grad(x):
        X = x.reshape(N, 3)
        X = _project(X)
        E, g = _energy_and_grad(X)
        return E, g.flatten()
    
    x_flat = _project(X0).flatten()
    
    while True:
        elapsed = time.time() - time_start
        if elapsed > TIME_LIMIT - 5:
            break
        
        result = minimize(
            obj_and_grad,
            x_flat,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 2000,
                'gtol': 1e-10,
                'ftol': 1e-14,
                'maxfun': 5000,
                'maxls': 50
            }
        )
        
        if result.success or result.status == 1:
            X_opt = _project(result.x.reshape(N, 3))
            E_opt, _ = _energy_and_grad(X_opt)
            if E_opt < best_E - 1e-12:
                best_X = X_opt
                best_E = E_opt
            x_flat = X_opt.flatten()
        else:
            # Try to still use the result
            X_opt = _project(result.x.reshape(N, 3))
            E_opt, _ = _energy_and_grad(X_opt)
            if E_opt < best_E - 1e-12:
                best_X = X_opt
                best_E = E_opt
                x_flat = X_opt.flatten()
            else:
                break
        
        # Check time
        elapsed = time.time() - time_start
        if elapsed > TIME_LIMIT - 10:
            break
        # If converged tightly, stop
        if result.success and result.fun < best_E + 1e-10:
            break
    
    return best_X, best_E

def construct():
    time_start = time.time()
    
    # Start with Fibonacci spiral
    X0 = _fibonacci_init()
    best_X, best_E = _optimize_from(X0, time_start, "fib")
    
    elapsed = time.time() - time_start
    if elapsed > TIME_LIMIT - 30:
        return _project(best_X)
    
    # Try a few random restarts
    for seed in [1, 7, 13, 42, 99]:
        elapsed = time.time() - time_start
        if elapsed > TIME_LIMIT - 20:
            break
        
        Xr = _random_init(seed)
        X_opt, E_opt = _optimize_from(Xr, time_start, f"rand_{seed}")
        if E_opt < best_E - 1e-10:
            best_X = X_opt
            best_E = E_opt
    
    # Final polish with tighter tolerance
    elapsed = time.time() - time_start
    if elapsed < TIME_LIMIT - 10:
        def obj_and_grad(x):
            X = x.reshape(N, 3)
            X = _project(X)
            E, g = _energy_and_grad(X)
            return E, g.flatten()
        
        x_flat = best_X.flatten()
        result = minimize(
            obj_and_grad,
            x_flat,
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': 5000,
                'gtol': 1e-12,
                'ftol': 1e-16,
                'maxfun': 10000,
                'maxls': 100
            }
        )
        X_polished = _project(result.x.reshape(N, 3))
        E_polished, _ = _energy_and_grad(X_polished)
        if E_polished < best_E - 1e-14:
            best_X = X_polished
            best_E = E_polished
    
    return _project(best_X)
