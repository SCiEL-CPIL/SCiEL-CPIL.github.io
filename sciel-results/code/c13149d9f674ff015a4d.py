"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize


def _log_energy(points):
    """Compute log(1/d) energy using math.fsum for accuracy."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    iu = np.triu_indices(n, k=1)
    log_dists = 0.5 * np.log(dists_sq[iu])
    return -math.fsum(log_dists.tolist())


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_uniform_sphere(n, seed=42):
    """Generate random uniform points on sphere."""
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _energy_and_grad(x):
    """Energy and gradient for L-BFGS-B. x is flat array of shape (N*3,)."""
    points = x.reshape(N, 3)
    points = _project_to_sphere(points)
    
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    
    # E = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    # dE/dx_i = sum_{j!=i} (x_j - x_i) / d_ij^2
    weights = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    weights[mask] = 1.0 / dists_sq[mask]
    
    sum_weights = np.sum(weights, axis=1)
    weighted_sum = weights @ points
    
    # grad_i = sum_j weights[i,j] * (x_j - x_i) = weighted_sum[i] - x_i * sum_weights[i]
    grads = weighted_sum - points * sum_weights[:, np.newaxis]
    
    # Project gradient to tangent space
    dot_products = np.sum(grads * points, axis=1, keepdims=True)
    grads_tangent = grads - dot_products * points
    
    energy = _log_energy(points)
    return energy, grads_tangent.flatten()


def _gradient_descent_sphere(points, lr=0.01, steps=5000, deadline=None):
    """Perform gradient descent on the sphere with adaptive step size."""
    best_points = points.copy()
    best_energy = _log_energy(best_points)
    
    for step in range(steps):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute gradient
        n = len(best_points)
        diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
        dists_sq = np.sum(diff * diff, axis=-1)
        np.fill_diagonal(dists_sq, 1.0)
        
        weights = np.zeros_like(dists_sq)
        mask = ~np.eye(n, dtype=bool)
        weights[mask] = 1.0 / dists_sq[mask]
        
        sum_weights = np.sum(weights, axis=1)
        weighted_sum = weights @ best_points
        grads = weighted_sum - best_points * sum_weights[:, np.newaxis]
        
        # Project to tangent space
        dot_products = np.sum(grads * best_points, axis=1, keepdims=True)
        grads_tangent = grads - dot_products * best_points
        
        # Adaptive step size
        grad_norm = np.linalg.norm(grads_tangent)
        if grad_norm < 1e-12:
            break
        
        new_points = _project_to_sphere(best_points - lr * grads_tangent)
        new_energy = _log_energy(new_points)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_points = new_points
            # Slowly decrease learning rate
            lr *= 0.999
        else:
            lr *= 0.5
    
    return best_points, best_energy


def _basin_hopping_step(points, energy, temperature, deadline=None):
    """Perform a single basin hopping step."""
    # Random perturbation
    n = len(points)
    rng = np.random.RandomState()
    perturbation = rng.randn(n, 3) * 0.1
    perturbed = _project_to_sphere(points + perturbation)
    
    # Local minimize
    x0 = perturbed.flatten()
    try:
        res = minimize(
            _energy_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 500, 'maxfun': 5000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        new_points = _project_to_sphere(res.x.reshape(N, 3))
        new_energy = _log_energy(new_points)
    except Exception:
        return points, energy
    
    # Metropolis acceptance
    if new_energy < energy or rng.random() < math.exp(-(new_energy - energy) / temperature):
        return new_points, new_energy
    
    return points, energy


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Multiple initializations
    inits = [
        _fibonacci_sphere(N),
        _random_uniform_sphere(N, seed=1),
        _random_uniform_sphere(N, seed=2),
        _random_uniform_sphere(N, seed=3),
        _random_uniform_sphere(N, seed=4),
        _random_uniform_sphere(N, seed=5),
    ]
    
    best_points = inits[0].copy()
    best_energy = _log_energy(best_points)
    
    # Phase 1: L-BFGS from multiple initializations
    for idx, init in enumerate(inits):
        if time.time() > deadline - 30.0:
            break
        
        x0 = init.flatten()
        
        try:
            res = minimize(
                _energy_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 2000, 'maxfun': 20000, 'ftol': 1e-12, 'gtol': 1e-9, 'maxls': 50}
            )
            
            if res.fun < best_energy:
                p_res = _project_to_sphere(res.x.reshape(N, 3))
                e_check = _log_energy(p_res)
                if e_check < best_energy:
                    best_energy = e_check
                    best_points = p_res.copy()
        except Exception:
            pass
    
    # Phase 2: Gradient descent refinement
    if time.time() < deadline - 20.0:
        gd_points, gd_energy = _gradient_descent_sphere(
            best_points, lr=0.005, steps=3000, deadline=deadline - 20.0
        )
        if gd_energy < best_energy:
            best_energy = gd_energy
            best_points = gd_points.copy()
    
    # Phase 3: Basin hopping for escape from local minima
    if time.time() < deadline - 10.0:
        temperature = 0.01
        for step in range(50):
            if time.time() > deadline - 10.0:
                break
            
            bh_points, bh_energy = _basin_hopping_step(
                best_points, best_energy, temperature, deadline=deadline - 10.0
            )
            
            if bh_energy < best_energy:
                best_energy = bh_energy
                best_points = bh_points.copy()
                temperature *= 0.95
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 5.0:
        x_best = best_points.flatten()
        try:
            res = minimize(
                _energy_and_grad,
                x_best,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 5000, 'maxfun': 50000, 'ftol': 1e-14, 'gtol': 1e-11, 'maxls': 50}
            )
            p_res = _project_to_sphere(res.x.reshape(N, 3))
            e_check = _log_energy(p_res)
            if e_check < best_energy:
                best_energy = e_check
                best_points = p_res.copy()
        except Exception:
            pass
    
    # Phase 5: Ultra-precise final polish
    if time.time() < deadline - 2.0:
        x_best = best_points.flatten()
        try:
            res = minimize(
                _energy_and_grad,
                x_best,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 3000, 'maxfun': 30000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
            )
            p_res = _project_to_sphere(res.x.reshape(N, 3))
            e_check = _log_energy(p_res)
            if e_check < best_energy:
                best_energy = e_check
                best_points = p_res.copy()
        except Exception:
            pass
    
    best_points = _project_to_sphere(best_points)
    
    if not np.all(np.isfinite(best_points)):
        return _fibonacci_sphere(N)
    
    # Check for points too close
    diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    iu = np.triu_indices(N, k=1)
    if np.min(dists[iu]) < 1e-9:
        return _fibonacci_sphere(N)
    
    return best_points
# EVOLVE-BLOCK-END
