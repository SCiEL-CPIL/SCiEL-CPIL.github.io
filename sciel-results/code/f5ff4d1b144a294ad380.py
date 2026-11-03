"""
Minimize logarithmic energy of 150 points on the unit sphere.
Uses L-BFGS-B on angular parameters with multiple restarts and a final polish.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
TIME_LIMIT = 118.0  # Safety margin before 120s kill


def _energy_and_grad(pos):
    """
    Compute logarithmic energy and gradient for points on unit sphere.
    pos: (N, 3) array of points on unit sphere.
    Returns (energy, grad) where grad is the unconstrained gradient w.r.t. pos.
    """
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    energy = -np.sum(np.log(d))
    
    inv_d2 = 1.0 / dist_sq
    np.fill_diagonal(inv_d2, 0.0)
    
    sum_pj_over_d2 = inv_d2 @ pos
    sum_inv_d2 = np.sum(inv_d2, axis=1)
    
    grad = sum_pj_over_d2 - pos * sum_inv_d2[:, np.newaxis]
    
    return energy, grad


def _project_to_sphere(pos):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pos, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pos / norms


def _params_to_sphere(theta):
    """Convert (theta, phi) parameters to points on the sphere.
    theta in [0, pi], phi in [0, 2*pi]
    """
    sin_t = np.sin(theta)
    x = sin_t * np.cos(phi)
    y = sin_t * np.sin(phi)
    z = np.cos(theta)
    return np.column_stack([x, y, z])


def _sphere_to_params(pos):
    """Convert points on sphere to (theta, phi) parameters."""
    x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
    theta = np.arccos(np.clip(z, -1.0, 1.0))
    phi = np.arctan2(y, x)
    # Ensure phi in [0, 2*pi]
    phi = np.where(phi < 0, phi + 2 * np.pi, phi)
    return np.column_stack([theta, phi])


def _energy_from_params(params):
    """Compute energy from (theta, phi) parameters."""
    pos = _params_to_sphere(params)
    energy, _ = _energy_and_grad(pos)
    return energy


def _grad_from_params(params):
    """Compute gradient w.r.t. parameters using chain rule."""
    pos = _params_to_sphere(params)
    energy, grad_pos = _energy_and_grad(pos)
    
    theta = params[:, 0]
    phi = params[:, 1]
    
    # d(pos)/d(theta): [cos(theta)*cos(phi), cos(theta)*sin(phi), -sin(theta)]
    # d(pos)/d(phi):   [-sin(phi), cos(phi), 0]
    
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    sin_p = np.sin(phi)
    cos_p = np.cos(phi)
    
    # grad_params[:, 0] = grad_pos . d(pos)/d(theta)
    # grad_params[:, 1] = grad_pos . d(pos)/d(phi)
    
    d_pos_d_theta = np.column_stack([cos_t * cos_p, cos_t * sin_p, -sin_t])
    d_pos_d_phi = np.column_stack([-sin_p, cos_p, np.zeros(N)])
    
    grad_theta = np.sum(grad_pos * d_pos_d_theta, axis=1)
    grad_phi = np.sum(grad_pos * d_pos_d_phi, axis=1)
    
    return energy, np.column_stack([grad_theta, grad_phi])


def _fibonacci_sphere():
    """Generate Fibonacci spiral points."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    pos = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(pos)


def _random_sphere(rng):
    """Generate random uniform points on sphere."""
    pos = rng.normal(0, 1, size=(N, 3))
    return _project_to_sphere(pos)


def _optimize_single(params_init, time_budget, rng):
    """
    Optimize a single starting configuration using L-BFGS-B on (theta, phi) params.
    """
    start_time = time.time()
    params = params_init.copy()
    
    # Initial energy
    energy, grad = _grad_from_params(params)
    best_params = params.copy()
    best_energy = energy
    
    # Bounds: theta in [0, pi], phi in [0, 2*pi]
    bounds = [(0.0, np.pi)] * N + [(0.0, 2 * np.pi)] * N
    
    # L-BFGS-B optimization
    try:
        result = minimize(
            _energy_from_params,
            params,
            jac=lambda p: _grad_from_params(p)[1],
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
        )
        
        if result.fun < best_energy:
            best_energy = result.fun
            best_params = result.x.copy()
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > time_budget:
        return _params_to_sphere(best_params), best_energy
    
    # Second polish with tighter tolerances if time permits
    if time_budget - elapsed > 10.0:
        try:
            result2 = minimize(
                _energy_from_params,
                best_params,
                jac=lambda p: _grad_from_params(p)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
            if result2.fun < best_energy:
                best_energy = result2.fun
                best_params = result2.x.copy()
        except Exception:
            pass
    
    return _params_to_sphere(best_params), best_energy


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Generate starting configurations
    starts = []
    
    # 1. Fibonacci spiral
    pos_fib = _fibonacci_sphere()
    params_fib = _sphere_to_params(pos_fib)
    starts.append(params_fib)
    
    # 2. Fibonacci with perturbation
    pos_fib_pert = pos_fib + rng.normal(0, 0.05, size=(N, 3))
    pos_fib_pert = _project_to_sphere(pos_fib_pert)
    params_fib_pert = _sphere_to_params(pos_fib_pert)
    starts.append(params_fib_pert)
    
    # 3-8. Random starts
    for _ in range(6):
        pos_rand = _random_sphere(rng)
        params_rand = _sphere_to_params(pos_rand)
        starts.append(params_rand)
    
    best_pos = _params_to_sphere(starts[0])
    best_energy = float('inf')
    
    # Time allocation
    total_opt_time = TIME_LIMIT - 8.0  # 8s buffer
    n_starts = len(starts)
    
    # Give more time to the first two (Fibonacci-based) starts
    time_weights = [0.25, 0.20] + [0.55 / (n_starts - 2)] * (n_starts - 2)
    
    for idx, start_params in enumerate(starts):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 5:
            break
        
        alloc_time = min(total_opt_time * time_weights[idx], TIME_LIMIT - 5 - elapsed)
        if alloc_time < 2.0:
            continue
        
        pos, energy = _optimize_single(start_params, alloc_time, rng)
        
        if energy < best_energy:
            best_energy = energy
            best_pos = pos.copy()
    
    # Final validation
    best_pos = _project_to_sphere(best_pos)
    
    if not np.all(np.isfinite(best_pos)):
        best_pos = _fibonacci_sphere()
    
    # Check minimum distance
    diff = best_pos[:, np.newaxis, :] - best_pos[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dist, 1.0)
    min_dist = np.min(dist)
    
    if min_dist < 1e-9:
        # Perturb and retry
        perturbed = best_pos + rng.normal(0, 0.01, size=(N, 3))
        perturbed = _project_to_sphere(perturbed)
        params_pert = _sphere_to_params(perturbed)
        
        remaining = TIME_LIMIT - 3 - (time.time() - start_time)
        if remaining > 5.0:
            pos, energy = _optimize_single(params_pert, min(remaining, 20.0), rng)
            if energy < best_energy:
                best_pos = pos
                best_energy = energy
            else:
                best_pos = perturbed
        else:
            best_pos = perturbed
    
    return best_pos
