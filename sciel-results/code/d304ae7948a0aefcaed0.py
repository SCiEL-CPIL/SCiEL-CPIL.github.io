"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import numpy as np
import time
import math
from scipy.optimize import minimize

# EVOLVE-BLOCK-START

def _energy_and_grad(X):
    """
    Compute the logarithmic energy and its gradient for points on the unit sphere.
    X: (N, 3) array of points.
    Returns: energy (float), grad (N, 3) array.
    """
    N = X.shape[0]
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    D2 = np.sum(diff**2, axis=2)
    
    # Avoid division by zero for i==j
    np.fill_diagonal(D2, 1.0)
    D = np.sqrt(D2)
    
    mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    log_D = np.log(D)
    energy = -np.sum(log_D[mask])
    
    inv_D2 = 1.0 / D2
    np.fill_diagonal(inv_D2, 0.0)
    
    grad = -np.sum(diff * inv_D2[:, :, np.newaxis], axis=1)
    
    return energy, grad

def _objective_and_grad(x_flat):
    """
    Objective and gradient for flat vector input.
    """
    X = x_flat.reshape(N, 3)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / norms
    
    energy, grad = _energy_and_grad(X)
    
    grad_dot_X = np.sum(grad * X, axis=1, keepdims=True)
    grad_tangent = grad - grad_dot_X * X
    
    grad_flat = grad_tangent.flatten()
    
    return energy, grad_flat

def _fibonacci_spiral():
    """Generate Fibonacci spiral initialization."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    return X

def _random_points(seed=None):
    """Generate random points on sphere."""
    if seed is not None:
        np.random.seed(seed)
    theta = np.random.uniform(0, 2 * np.pi, N)
    phi = np.arccos(2 * np.random.uniform(0, 1, N) - 1)
    X = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])
    return X

def _lbfgs_optimize(X0, time_budget, ftol=1e-15, gtol=1e-12, maxiter=2000):
    """
    Optimize from a starting point using L-BFGS-B with tangent-space gradient.
    Returns best X found.
    """
    start_time = time.time()
    x0 = X0.flatten()
    
    result = minimize(
        _objective_and_grad,
        x0,
        method='L-BFGS-B',
        jac=True,
        options={
            'maxiter': maxiter,
            'ftol': ftol,
            'gtol': gtol,
            'maxls': 50,
            'maxfun': 5000
        }
    )
    
    X_opt = result.x.reshape(N, 3)
    X_opt = X_opt / np.linalg.norm(X_opt, axis=1, keepdims=True)
    
    elapsed = time.time() - start_time
    return X_opt, elapsed

def _basin_hopping_step(X, step_size, time_budget):
    """
    Perform a single basin hopping step: perturb and optimize.
    Returns the improved X or None if no improvement.
    """
    # Generate random perturbation on tangent space
    noise = np.random.normal(0, step_size, (N, 3))
    # Project noise to tangent space
    noise = noise - np.sum(noise * X, axis=1, keepdims=True) * X
    X_perturbed = X + noise
    X_perturbed = X_perturbed / np.linalg.norm(X_perturbed, axis=1, keepdims=True)
    
    # Optimize from perturbed point
    X_opt, elapsed = _lbfgs_optimize(X_perturbed, time_budget)
    
    if not np.all(np.isfinite(X_opt)):
        return None
    
    # Check minimum distance
    diff = X_opt[:, np.newaxis, :] - X_opt[np.newaxis, :, :]
    D = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(D, np.inf)
    min_dist = np.min(D)
    
    if min_dist < 1e-9:
        return None
    
    energy, _ = _energy_and_grad(X_opt)
    return X_opt, energy

def _high_precision_polish(X, time_budget):
    """
    Final ultra-tight L-BFGS polish pass.
    """
    X_opt, elapsed = _lbfgs_optimize(X, time_budget, ftol=1e-16, gtol=1e-14, maxiter=5000)
    if np.all(np.isfinite(X_opt)):
        X_opt = X_opt / np.linalg.norm(X_opt, axis=1, keepdims=True)
        return X_opt
    return X

def construct():
    start_time = time.time()
    time_limit = 115.0  # Leave 35s margin for evaluation overhead
    
    best_X = None
    best_energy = np.inf
    
    def _update_best(X):
        nonlocal best_X, best_energy
        if not np.all(np.isfinite(X)):
            return
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        D = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(D, np.inf)
        if np.min(D) < 1e-9:
            return
        energy, _ = _energy_and_grad(X)
        if energy < best_energy:
            best_energy = energy
            best_X = X.copy()
    
    # Phase 1: Multi-start L-BFGS optimization
    starts = []
    starts.append(_fibonacci_spiral())
    for seed in [42, 123, 456, 789, 101, 202, 303, 404]:
        starts.append(_random_points(seed))
    
    # Perturbed Fibonacci
    X_fib = _fibonacci_spiral()
    for noise_scale in [0.05, 0.1, 0.2]:
        np.random.seed(999)
        noise = np.random.normal(0, noise_scale, (N, 3))
        X_perturbed = X_fib + noise
        X_perturbed = X_perturbed / np.linalg.norm(X_perturbed, axis=1, keepdims=True)
        starts.append(X_perturbed)
    
    for i, X0 in enumerate(starts):
        if time.time() - start_time > time_limit * 0.5:
            break
        try:
            X_opt, elapsed = _lbfgs_optimize(X0, time_limit)
            _update_best(X_opt)
        except Exception:
            continue
    
    # Phase 2: Basin hopping with L-BFGS polish
    if best_X is not None:
        current_X = best_X.copy()
        current_energy = best_energy
        step_size = 0.3
        
        for bh_iter in range(50):
            if time.time() - start_time > time_limit * 0.85:
                break
            
            remaining = time_limit - (time.time() - start_time)
            if remaining < 10.0:
                break
            
            # Try multiple perturbations
            found_improvement = False
            for attempt in range(3):
                if time.time() - start_time > time_limit * 0.85:
                    break
                
                result = _basin_hopping_step(current_X, step_size, remaining)
                if result is not None:
                    X_new, E_new = result
                    if E_new < current_energy:
                        current_X = X_new
                        current_energy = E_new
                        _update_best(X_new)
                        found_improvement = True
                        break
            
            # Adapt step size
            if found_improvement:
                step_size = max(0.05, step_size * 0.8)
            else:
                step_size = min(1.0, step_size * 1.3)
    
    # Phase 3: Final high-precision polish
    if best_X is not None:
        if time.time() - start_time < time_limit - 5.0:
            X_polished = _high_precision_polish(best_X, time_limit - (time.time() - start_time))
            _update_best(X_polished)
    
    # Fallback
    if best_X is None:
        best_X = _fibonacci_spiral()
    
    return best_X

# EVOLVE-BLOCK-END
