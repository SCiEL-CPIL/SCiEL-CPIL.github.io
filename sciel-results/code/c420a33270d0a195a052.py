"""Rewritten program for minimal logarithmic energy of 100 points on the sphere."""
import numpy as np
from scipy.optimize import minimize
import time
import math

N = 100
TIME_LIMIT = 140.0  # Safety margin before 150s kill


def _log_energy(x):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)"""
    n = x.shape[0]
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(n, k=1)
    d = dist[iu]
    return -np.sum(np.log(d))


def _log_energy_grad(x):
    """Compute gradient of logarithmic energy with respect to x.
    
    Energy E = -sum_{i<j} log(|x_i - x_j|)
    dE/dx_i = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    """
    n = x.shape[0]
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.sum(diff * diff, axis=2)  # (n, n)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    coeff = 1.0 / dist_sq  # (n, n)
    np.fill_diagonal(coeff, 0.0)
    
    # grad[i] = sum_{j != i} (x_j - x_i)/|x_i-x_j|^2 = -sum_{j != i} diff[i,j]/dist_sq[i,j]
    grad = -np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    
    return grad


def _project_to_sphere(x):
    """Project points back to unit sphere."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return x / norms


def _golden_spiral(n, seed_offset=0.0):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * (i + seed_offset)
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere_points(n, rng):
    """Generate random points on the unit sphere."""
    v = rng.standard_normal((n, 3))
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / norms


def _tangent_project(grad, x):
    """Project gradient to tangent space of sphere at each point."""
    return grad - np.sum(grad * x, axis=1, keepdims=True) * x


def _optimize_lbfgs(x0, time_budget, maxiter=2000, ftol=1e-14, gtol=1e-10):
    """Run L-BFGS-B optimization from x0 with a time budget."""
    start_time = time.time()
    
    def objective_only(x_flat):
        x = x_flat.reshape(-1, 3)
        x = _project_to_sphere(x)
        return _log_energy(x)
    
    def objective_and_grad(x_flat):
        x = x_flat.reshape(-1, 3)
        x = _project_to_sphere(x)
        e = _log_energy(x)
        g = _log_energy_grad(x)
        g_tangent = _tangent_project(g, x)
        return e, g_tangent.flatten()
    
    x_flat = x0.flatten()
    
    best_x = x0.copy()
    best_e = _log_energy(x0)
    
    try:
        res = minimize(
            lambda xf: objective_only(xf),
            x_flat,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'maxfun': maxiter * 2, 'ftol': ftol, 'gtol': gtol}
        )
        x_opt = _project_to_sphere(res.x.reshape(-1, 3))
        e_opt = _log_energy(x_opt)
        if e_opt < best_e:
            best_e = e_opt
            best_x = x_opt.copy()
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed < time_budget:
        # Try one more refinement pass
        try:
            res = minimize(
                lambda xf: objective_only(xf),
                best_x.flatten(),
                method='L-BFGS-B',
                options={'maxiter': maxiter, 'maxfun': maxiter * 2, 'ftol': ftol * 0.1, 'gtol': gtol * 0.1}
            )
            x_opt = _project_to_sphere(res.x.reshape(-1, 3))
            e_opt = _log_energy(x_opt)
            if e_opt < best_e:
                best_e = e_opt
                best_x = x_opt.copy()
        except Exception:
            pass
    
    return best_x, best_e


def _optimize_nelder_mead(x0, time_budget):
    """Run Nelder-Mead optimization from x0 with a time budget."""
    start_time = time.time()
    
    def objective_only(x_flat):
        x = x_flat.reshape(-1, 3)
        x = _project_to_sphere(x)
        return _log_energy(x)
    
    x_flat = x0.flatten()
    
    best_x = x0.copy()
    best_e = _log_energy(x0)
    
    try:
        res = minimize(
            objective_only,
            x_flat,
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-12, 'adaptive': True}
        )
        x_opt = _project_to_sphere(res.x.reshape(-1, 3))
        e_opt = _log_energy(x_opt)
        if e_opt < best_e:
            best_e = e_opt
            best_x = x_opt.copy()
    except Exception:
        pass
    
    return best_x, best_e


def _local_rearrange(x, time_budget):
    """Try local rearrangement by swapping pairs of points to reduce energy."""
    start_time = time.time()
    x = x.copy()
    e = _log_energy(x)
    n = x.shape[0]
    
    # Precompute distances
    def compute_energy(pts):
        return _log_energy(pts)
    
    improved = True
    while improved and (time.time() - start_time) < time_budget:
        improved = False
        for i in range(n):
            if (time.time() - start_time) >= time_budget:
                break
            for j in range(i + 1, n):
                if (time.time() - start_time) >= time_budget:
                    break
                # Try swapping
                x_swap = x.copy()
                x_swap[i], x_swap[j] = x[j], x[i]
                e_swap = compute_energy(x_swap)
                if e_swap < e - 1e-12:
                    x = x_swap
                    e = e_swap
                    improved = True
    
    return x, e


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    inits = []
    
    # Golden spiral
    inits.append(_golden_spiral(N))
    
    # Perturbed golden spiral
    for scale in [0.1, 0.05, 0.02]:
        perturbed = _golden_spiral(N) + scale * rng.standard_normal((N, 3))
        inits.append(_project_to_sphere(perturbed))
    
    # Random points
    for _ in range(5):
        inits.append(_random_sphere_points(N, rng))
    
    # Different golden spiral offsets
    for offset in [1.0, 2.0, 0.5, 3.0, 1.5]:
        inits.append(_golden_spiral(N, seed_offset=offset))
    
    best_x = None
    best_e = float('inf')
    
    num_inits = len(inits)
    
    for i, x0 in enumerate(inits):
        elapsed = time.time() - start_time
        remaining = TIME_LIMIT - elapsed
        if remaining < 15:
            break
        
        # Allocate time budget for this optimization
        time_budget = max(5.0, remaining / max(1, num_inits - i))
        
        # First try L-BFGS-B
        x_opt, e_opt = _optimize_lbfgs(x0, time_budget * 0.6, maxiter=1500, ftol=1e-13, gtol=1e-10)
        
        if e_opt < best_e:
            best_e = e_opt
            best_x = x_opt.copy()
        
        # Check if we're running low on time
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 20:
            break
        
        # If we have time, try Nelder-Mead from the best result so far
        remaining = TIME_LIMIT - elapsed
        if remaining > 30:
            nm_budget = min(remaining * 0.5, 40.0)
            x_nm, e_nm = _optimize_nelder_mead(x_opt, nm_budget)
            if e_nm < best_e:
                best_e = e_nm
                best_x = x_nm.copy()
        
        # Check if we're running low on time
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 10:
            break
    
    if best_x is None:
        best_x = _golden_spiral(N)
        best_e = _log_energy(best_x)
    
    # Final projection to ensure unit sphere
    best_x = _project_to_sphere(best_x)
    
    # Final refinement pass with very tight tolerances if time allows
    elapsed = time.time() - start_time
    remaining = TIME_LIMIT - elapsed
    if remaining > 20:
        x_final, e_final = _optimize_lbfgs(best_x, remaining * 0.8, maxiter=3000, ftol=1e-15, gtol=1e-12)
        if e_final < best_e:
            best_e = e_final
            best_x = x_final.copy()
    
    # Verify no points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_x[i] - best_x[j])
            if d < 1e-9:
                # Nudge point j slightly
                best_x[j] = best_x[j] + 1e-8 * rng.standard_normal(3)
                best_x[j] = _project_to_sphere(best_x[j])
    
    return best_x
