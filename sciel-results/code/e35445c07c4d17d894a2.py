"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math

def _energy_and_grad(X):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    Gradient with respect to x_i (projected to tangent space):
      dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[0]
    # Compute pairwise distances
    # d_ij = |x_i - x_j|
    # Use broadcasting: D[i,j] = |x_i - x_j|
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # (n, n, 3)
    dist = np.sqrt(np.sum(diff * diff, axis=2))  # (n, n)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum over i<j of log(1/dist_ij) = -sum over i<j of log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    d_upper = dist[iu]
    E = -np.sum(np.log(d_upper))
    
    # Gradient: dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # grad[i] = sum_j (X[i] - X[j]) / dist[i,j]^2
    # diff[i,j] = X[i] - X[j]
    # grad[i] = sum_j diff[i,j] / dist[i,j]^2
    inv_d2 = 1.0 / (dist * dist)  # (n, n)
    np.fill_diagonal(inv_d2, 0.0)
    grad = np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)  # (n, 3)
    
    return E, grad

def _project_to_sphere(X):
    """Normalize rows of X to unit length."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return X / norms

def _fibonacci_sphere(n):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed):
    """Generate n random points on unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _project_to_sphere(pts)

def _optimize_from_start(X0, time_budget):
    """Run L-BFGS-B optimization starting from X0, with time budget in seconds."""
    X0 = _project_to_sphere(X0.copy())
    
    # Use L-BFGS-B with gradient
    def fun(X):
        E, g = _energy_and_grad(X)
        return E, g
    
    try:
        res = minimize(
            fun, X0, method='L-BFGS-B', jac=True,
            options={'maxiter': 5000, 'gtol': 1e-12, 'maxls': 50}
        )
        X_opt = _project_to_sphere(res.x)
        E_opt = _energy_and_grad(X_opt)[0]
        return X_opt, E_opt
    except Exception:
        return X0, _energy_and_grad(X0)[0]

def _polish(X, time_budget):
    """High-precision polish using gradient descent with adaptive step size."""
    X = _project_to_sphere(X.copy())
    E_prev = _energy_and_grad(X)[0]
    
    # Adaptive gradient descent with projection
    lr = 1e-3
    min_lr = 1e-8
    patience = 0
    max_patience = 200
    
    while time_budget > 0.5 and patience < max_patience:
        E, g = _energy_and_grad(X)
        if not np.isfinite(E):
            break
        
        # Project gradient to tangent space
        g_tan = g - np.sum(g * X, axis=1, keepdims=True) * X
        
        # Adaptive step: try to reduce energy
        X_new = X + lr * g_tan
        X_new = _project_to_sphere(X_new)
        E_new = _energy_and_grad(X_new)[0]
        
        if E_new < E_prev - 1e-12:
            X = X_new
            E_prev = E_new
            patience = 0
            # Slightly increase lr if improving well
            if patience == 0 and lr < 1e-2:
                lr = min(lr * 1.1, 1e-2)
        else:
            patience += 1
            lr *= 0.8
            if lr < min_lr:
                break
    
    return X, _energy_and_grad(X)[0]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Multiple diverse starting points
    starts = []
    
    # 1. Fibonacci spiral
    starts.append(_fibonacci_sphere(N))
    
    # 2. Random starts with different seeds
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        starts.append(_random_sphere(N, seed))
    
    # 3. Anisotropic scaling variations of Fibonacci
    for scale in [0.8, 1.2, 0.6, 1.5]:
        Xf = _fibonacci_sphere(N)
        Xs = Xf.copy()
        Xs[:, 0] *= scale
        starts.append(_project_to_sphere(Xs))
    
    # 4. Spherical cap perturbations
    Xf = _fibonacci_sphere(N)
    rng = np.random.RandomState(999)
    noise = rng.randn(N, 3) * 0.1
    starts.append(_project_to_sphere(Xf + noise))
    
    best_X = None
    best_E = np.inf
    
    for i, X0 in enumerate(starts):
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 5.0:
            break
        
        # Allocate time for optimization
        opt_time = min(remaining - 2.0, 30.0)
        if opt_time < 2.0:
            break
        
        X_opt, E_opt = _optimize_from_start(X0, opt_time)
        
        if E_opt < best_E:
            best_E = E_opt
            best_X = X_opt.copy()
        
        # Check time
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 3.0:
            break
    
    # Polish the best result if time permits
    if best_X is not None:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining > 5.0:
            best_X, best_E = _polish(best_X, remaining - 2.0)
    
    # Final check
    if best_X is None:
        best_X = _fibonacci_sphere(N)
    
    # Ensure all points are on unit sphere and valid
    best_X = _project_to_sphere(best_X)
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_X[i] - best_X[j])
            if d < 1e-9:
                # Perturb slightly
                rng = np.random.RandomState(i*1000+j)
                best_X[j] += rng.randn(3) * 1e-8
                best_X[j] = _project_to_sphere(best_X[j])
    
    return best_X
# EVOLVE-BLOCK-END
