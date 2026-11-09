"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 150

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import math


def _fibonacci_points(n, seed_offset=0.0):
    """Generate Fibonacci spiral points on the sphere."""
    i = np.arange(n) + 0.5 + seed_offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(n):
    """Generate random points on the sphere."""
    pts = np.random.normal(0, 1, size=(n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _normalize(pts):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms


def _log_energy(pts):
    """Compute sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    pair_dists = np.clip(pair_dists, 1e-15, None)
    return -np.sum(np.log(pair_dists))


def _log_energy_grad(pts):
    """Compute energy and gradient of log energy w.r.t. points in R^3.
    
    E = -sum_{i<j} log(||x_i - x_j||)
    dE/dx_i = -sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    """
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    
    inv_d2 = 1.0 / (dists * dists)
    np.fill_diagonal(inv_d2, 0.0)
    
    # grad[i] = sum_j (x_i - x_j) / d_ij^2
    grad_raw = (diff * inv_d2[:, :, np.newaxis]).sum(axis=1)
    
    # The actual gradient of E is -grad_raw
    grad_E = -grad_raw
    
    # Project gradient onto tangent space of sphere at each point
    # Tangent projection: g_tangent = g - (g . x) x
    dot = np.sum(grad_E * pts, axis=1, keepdims=True)
    grad_tangent = grad_E - dot * pts
    
    energy = _log_energy(pts)
    return energy, grad_tangent


def _spherical_lbfgs(pts0, max_time=5.0):
    """Optimize points on sphere using L-BFGS-B with spherical projection.
    
    We use a wrapper that projects back to sphere after each step.
    """
    pts = pts0.copy()
    t_start = time.time()
    
    def objective_and_grad(x_flat):
        pts_local = x_flat.reshape(-1, 3)
        # Project to sphere to ensure validity
        pts_local = _normalize(pts_local)
        energy, grad = _log_energy_grad(pts_local)
        # Flatten gradient
        grad_flat = grad.flatten()
        return energy, grad_flat
    
    # Use L-BFGS-B with tight tolerances
    try:
        res = minimize(
            objective_and_grad,
            pts.flatten(),
            method='L-BFGS-B',
            options={
                'maxiter': 200,
                'ftol': 1e-14,
                'gtol': 1e-8,
                'maxfun': 500,
            }
        )
        
        if res.success or res.fun < _log_energy(pts):
            pts_opt = _normalize(res.x.reshape(-1, 3))
            energy_opt = _log_energy(pts_opt)
            return pts_opt, energy_opt
        else:
            return pts, _log_energy(pts)
    
    except Exception:
        return pts, _log_energy(pts)


def _gradient_descent_sphere(pts0, max_time=5.0):
    """Fallback: gradient descent with momentum and re-projection."""
    pts = pts0.copy()
    lr = 0.02
    momentum = 0.8
    v = np.zeros_like(pts)
    t_start = time.time()
    
    while time.time() - t_start < max_time:
        energy, grad = _log_energy_grad(pts)
        v = momentum * v - lr * grad
        new_pts = _normalize(pts + v)
        new_energy = _log_energy(new_pts)
        
        if new_energy < energy:
            pts = new_pts
            energy = new_energy
        else:
            lr *= 0.5
            v *= 0.5
            if lr < 1e-8:
                break
    
    return pts, _log_energy(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    # Generate diverse initializations
    inits = []
    
    # 1. Fibonacci spiral (standard)
    inits.append(_fibonacci_points(N))
    
    # 2. Fibonacci with offset
    inits.append(_fibonacci_points(N, seed_offset=0.1))
    
    # 3-8. Random initializations (6 diverse starts)
    for _ in range(6):
        inits.append(_random_points(N))
    
    # 9. Clustered initialization: 3 clusters of 50
    cluster_centers = np.random.normal(0, 1, size=(3, 3))
    cluster_centers = cluster_centers / np.linalg.norm(cluster_centers, axis=1, keepdims=True)
    clustered = []
    for c in cluster_centers:
        local = c + np.random.normal(0, 0.3, size=(50, 3))
        clustered.append(local)
    inits.append(_normalize(np.vstack(clustered)))
    
    best_pts = inits[0]
    best_energy = _log_energy(best_pts)
    
    # Phase 1: Quick screening with gradient descent (fast)
    # Allocate ~30s for screening all inits
    screen_time_per_init = min(3.0, (deadline - start_time - 60.0) / len(inits))
    
    screened = []
    for i, init in enumerate(inits):
        if time.time() > deadline - 60.0:
            break
        try:
            opt_pts, opt_energy = _gradient_descent_sphere(init, max_time=screen_time_per_init)
            screened.append((opt_energy, opt_pts, i))
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts
        except Exception:
            pass
    
    # Phase 2: Refine top candidates with L-BFGS-B (slower but more precise)
    # Sort by energy and take top 3
    screened.sort(key=lambda x: x[0])
    top_candidates = screened[:3] if len(screened) >= 3 else screened
    
    lbfgs_time_per = min(20.0, (deadline - start_time - 10.0) / max(1, len(top_candidates)))
    
    for energy, pts, idx in top_candidates:
        if time.time() > deadline - 10.0:
            break
        try:
            opt_pts, opt_energy = _spherical_lbfgs(pts, max_time=lbfgs_time_per)
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts
        except Exception:
            pass
    
    # Phase 3: Final polish on best point with tighter L-BFGS
    if time.time() < deadline - 5.0:
        try:
            remaining = deadline - time.time() - 5.0
            opt_pts, opt_energy = _spherical_lbfgs(best_pts, max_time=remaining)
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts
        except Exception:
            pass
    
    # Final validation
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_points(N)
    
    # Ensure no points are too close
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(N, k=1)
    pair_dists = dists[iu]
    if np.min(pair_dists) < 1e-9:
        # Fallback to Fibonacci
        return _fibonacci_points(N)
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
