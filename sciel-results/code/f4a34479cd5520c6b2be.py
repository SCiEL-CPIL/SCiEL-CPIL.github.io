"""Rewritten program: deterministic gradient-based local minimization on the sphere
with multiple restarts and a deadline guard."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE_SECONDS = 145.0  # safety margin under 150s wall-clock


def _energy_and_grad(X):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i-x_j|) and its gradient w.r.t. X.
    X is (N,3) on the unit sphere. Returns (energy, grad) where grad is the
    Euclidean gradient (not yet projected)."""
    # Pairwise distances
    diff = X[:, None, :] - X[None, :, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)   # (N, N)
    # Avoid log(0) and division by zero
    np.fill_diagonal(dist, 1.0)
    # Energy: sum_{i<j} log(1/d) = -sum_{i<j} log(d)
    # Use upper triangle
    iu = np.triu_indices(N, k=1)
    d_ij = dist[iu]
    e = -np.sum(np.log(d_ij))
    
    # Gradient: d/dx_i [-log|x_i - x_j|] = (x_i - x_j)/|x_i - x_j|^2
    # For each i, grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # = sum_{j != i} (x_i - x_j) / dist[i,j]^2
    # Note: for j != i, (x_i - x_j)/dist^2
    inv_dist2 = np.zeros_like(dist)
    mask = ~np.eye(N, dtype=bool)
    inv_dist2[mask] = 1.0 / (dist[mask] ** 2)
    
    # grad[i] = sum_j (X[i] - X[j]) * inv_dist2[i,j]
    # = X[i] * sum_j inv_dist2[i,j] - sum_j X[j] * inv_dist2[i,j]
    row_sum = np.sum(inv_dist2, axis=1)  # (N,)
    # sum_j X[j] * inv_dist2[i,j] = inv_dist2 @ X
    weighted_sum = inv_dist2 @ X  # (N, 3)
    grad = X * row_sum[:, None] - weighted_sum
    
    return e, grad


def _project_to_sphere(X):
    """Normalize each row of X to unit length."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return X / norms


def _fibonacci_sphere(n, seed_offset=0.0):
    """Generate n points on the unit sphere using Fibonacci spiral with offset."""
    i = np.arange(n) + 0.5 + seed_offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Generate n random points on the unit sphere."""
    v = rng.normal(size=(n, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return v / norms


def _sobol_sphere(n, rng):
    """Generate n points on the unit sphere using Sobol sequence (quasi-random)."""
    from scipy.stats import qmc
    sampler = qmc.Sobol(d=3, scramble=True, seed=rng)
    points = sampler.random(n)
    u, v, w = points[:,0], points[:,1], points[:,2]
    theta = 2.0 * np.pi * u
    z = 2.0 * v - 1.0
    r = np.sqrt(np.maximum(0.0, 1.0 - z*z))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y, z])


def _penalized_energy_and_grad(X_flat, lambda_penalty=1000.0):
    """
    Objective and gradient for scipy minimize with a penalty for deviating from the sphere.
    X_flat is the flattened (N, 3) array.
    """
    X = X_flat.reshape(N, 3)
    e, grad = _energy_and_grad(X)
    
    # Penalty term: lambda * sum(||x_i||^2 - 1)^2
    norms_sq = np.sum(X**2, axis=1)
    penalty = lambda_penalty * np.sum((norms_sq - 1.0)**2)
    grad_penalty = 4.0 * lambda_penalty * (norms_sq - 1.0)[:, None] * X
    
    return e + penalty, grad + grad_penalty


def construct():
    start_time = time.time()
    
    best_X = None
    best_energy = float('inf')
    
    # Generate multiple initial configurations
    rng = np.random.default_rng(42)
    
    # Start with Fibonacci spiral (known good starting point)
    initial_configs = [_fibonacci_sphere(N)]
    
    # Add random perturbations of Fibonacci
    for _ in range(3):
        X0 = _fibonacci_sphere(N)
        noise = rng.normal(scale=0.1, size=X0.shape)
        X0 = _project_to_sphere(X0 + noise)
        initial_configs.append(X0)
    
    # Add fully random configurations
    for _ in range(3):
        initial_configs.append(_random_sphere(N, rng))
        
    # Add Sobol configurations
    initial_configs.append(_sobol_sphere(N, rng))
    
    # Add scaled Fibonacci with different offsets
    for offset in [0.1, 0.25, 0.4]:
        initial_configs.append(_fibonacci_sphere(N, seed_offset=offset))
    
    # Deadline for individual optimizations: leave 5s for final polish
    deadline_for_opt = start_time + DEADLINE_SECONDS - 5.0
    
    for X0 in initial_configs:
        if time.time() - start_time > deadline_for_opt:
            break
        
        x0_flat = X0.flatten()
        
        try:
            res = minimize(
                _penalized_energy_and_grad,
                x0_flat,
                method='L-BFGS-B',
                options={
                    'maxiter': 5000,
                    'ftol': 1e-12,
                    'gtol': 1e-10,
                    'maxfun': 10000,
                }
            )
        except Exception:
            continue
        
        X_opt = _project_to_sphere(res.x.reshape(N, 3))
        e_opt, _ = _energy_and_grad(X_opt)
        
        if e_opt < best_energy:
            best_energy = e_opt
            best_X = X_opt.copy()
    
    # Final polish: run a tighter optimization on the best solution found
    if best_X is not None and time.time() - start_time < DEADLINE_SECONDS - 1:
        try:
            x0_flat = best_X.flatten()
            res = minimize(
                _penalized_energy_and_grad,
                x0_flat,
                method='L-BFGS-B',
                options={
                    'maxiter': 10000,
                    'ftol': 1e-14,
                    'gtol': 1e-12,
                    'maxfun': 20000,
                }
            )
            X_polish = _project_to_sphere(res.x.reshape(N, 3))
            e_polish, _ = _energy_and_grad(X_polish)
            if e_polish < best_energy:
                best_energy = e_polish
                best_X = X_polish.copy()
        except Exception:
            pass
    
    if best_X is None:
        # Fallback: return Fibonacci spiral
        best_X = _fibonacci_sphere(N)
    
    return best_X
