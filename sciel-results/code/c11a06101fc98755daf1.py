"""Logarithmic energy minimization for 100 points on the unit sphere.
Uses repulsive pre-spread, multi-start L-BFGS-B in tangent space, and
high-precision coordinate polish."""

import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100
_DEADLINE = 145.0  # safety margin from 150s limit


def _project(X):
    """Normalize points to unit sphere."""
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _energy_and_grad(pts):
    """Compute log energy and tangent-space gradient for points on unit sphere.
    Energy: sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    """
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-30)
    dist = np.sqrt(dist_sq)
    
    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))
    
    # Gradient of E = -sum log(d_ij)
    # dE/dx_i = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    sum_W = W.sum(axis=1, keepdims=True)
    Wx = W @ pts
    raw_grad = sum_W * pts - Wx  # sum (x_i - x_j)/d^2
    grad = -raw_grad  # true gradient
    
    # Project onto tangent space: grad - (grad . x_i) x_i
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return energy, grad


def _precise_energy(pts):
    """Compute energy using math.fsum for maximum accuracy."""
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _fibonacci_spiral(n, offset=0.5):
    """Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, rng):
    """Random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    return _project(pts)


def _sobol_sphere(n, seed=0):
    """Generate points on sphere using Sobol sequence (low discrepancy)."""
    sampler_2d = qmc.Sobol(d=2, scramble=True, seed=seed)
    m = int(2 ** np.ceil(np.log2(n)))
    y = sampler_2d.random(m)
    
    theta = np.arccos(1 - 2 * y[:, 0])
    phi = 2 * np.pi * y[:, 1]
    
    x = np.sin(theta) * np.cos(phi)
    y_coord = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    pts = np.column_stack([x, y_coord, z])
    return pts[:n]


def _repulsive_prespread(pts, n_steps=200, strength=1.0, time_budget=2.0):
    """Apply repulsive forces to spread out points, avoiding clusters."""
    start = time.time()
    pts = _project(pts.copy())
    n = pts.shape[0]
    
    for step in range(n_steps):
        if time.time() - start > time_budget:
            break
        # Compute repulsive forces
        sq_norms = np.einsum('ij,ij->i', pts, pts)
        dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
        np.fill_diagonal(dist_sq, 1.0)
        dist_sq = np.maximum(dist_sq, 1e-20)
        dist = np.sqrt(dist_sq)
        
        # Repulsive force: (x_i - x_j) / d^3 (inverse cube law for stronger repulsion)
        W = 1.0 / (dist_sq * dist)
        np.fill_diagonal(W, 0.0)
        
        forces = (W @ pts) - W.sum(axis=1, keepdims=True) * pts
        
        # Normalize forces
        f_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        f_norms = np.where(f_norms < 1e-12, 1.0, f_norms)
        forces = forces / f_norms
        
        # Small step
        lr = 0.01 * strength / (1.0 + step * 0.01)
        pts = pts + lr * forces
        pts = _project(pts)
    
    return pts


def _tangent_to_sphere(pts, tvec):
    """Map tangent vectors to sphere via exponential map."""
    tv = tvec.reshape(N, 3)
    # Project to tangent space
    dots = np.sum(tv * pts, axis=1)
    tv = tv - dots[:, np.newaxis] * pts
    
    # Use exponential map
    tv_norm = np.linalg.norm(tv, axis=1, keepdims=True)
    safe_norm = np.where(tv_norm < 1e-12, 1.0, tv_norm)
    tv_unit = tv / safe_norm
    
    cos_t = np.cos(tv_norm)
    sin_t_over_norm = np.sin(tv_norm) / safe_norm
    
    new_pts = cos_t * pts + sin_t_over_norm * tv
    return _project(new_pts)


def _lbfgs_optimize(pts0, max_iter=3000, ftol=1e-15, gtol=1e-12):
    """Run L-BFGS-B in tangent space parameterization."""
    n = pts0.shape[0]
    
    def _obj_and_grad(tvec):
        pts = _tangent_to_sphere(pts0, tvec)
        e, g = _energy_and_grad(pts)
        return e, g.reshape(-1)
    
    try:
        res = minimize(_obj_and_grad, np.zeros(3 * n), method='L-BFGS-B',
                       jac=True, options={'maxiter': max_iter, 'ftol': ftol, 
                                          'gtol': gtol, 'maxls': 100, 'maxfun': 20000,
                                          'iprint': -1})
        pts_opt = _tangent_to_sphere(pts0, res.x)
        return pts_opt, res.fun
    except Exception:
        return pts0, np.inf


def _high_precision_polish(pts, max_rounds=200, time_budget=5.0):
    """High-precision coordinate descent polish with adaptive step sizes."""
    start = time.time()
    pts = _project(pts.copy())
    n = pts.shape[0]
    best_e = _precise_energy(pts)
    
    for rnd in range(max_rounds):
        if time.time() - start > time_budget:
            break
        improved = False
        for i in range(n):
            # Compute gradient for point i
            dists = np.linalg.norm(pts[i] - pts, axis=1)
            dists[i] = 1.0
            dists = np.maximum(dists, 1e-12)
            
            W = 1.0 / (dists ** 2)
            W[i] = 0.0
            raw_g = W.sum() * pts[i] - (W * pts).sum(axis=0)
            
            dot = np.dot(raw_g, pts[i])
            g_tan = raw_g - dot * pts[i]
            
            g_norm = np.linalg.norm(g_tan)
            if g_norm < 1e-15:
                continue
            
            # Adaptive step: try multiple step sizes
            best_step_e = best_e
            best_step = 0.0
            for step in [0.1, 0.05, 0.02, 0.01, 0.005, 0.002]:
                new_pt = pts[i] + step * g_tan
                new_pt = new_pt / np.linalg.norm(new_pt)
                pts_old_i = pts[i].copy()
                pts[i] = new_pt
                e = _precise_energy(pts)
                if e < best_step_e - 1e-13:
                    best_step_e = e
                    best_step = step
                pts[i] = pts_old_i
            
            if best_step > 0:
                pts[i] = pts[i] + best_step * g_tan
                pts[i] = pts[i] / np.linalg.norm(pts[i])
                best_e = best_step_e
                improved = True
        
        if not improved:
            break
    
    return _project(pts), best_e


def _local_optimize(pts, time_budget=10.0):
    """Local optimization with L-BFGS + high-precision polish."""
    start = time.time()
    pts = _project(pts.copy())
    best_pts = pts.copy()
    best_energy = _precise_energy(best_pts)
    
    # Phase 1: L-BFGS optimization
    try:
        pts_opt, e_opt = _lbfgs_optimize(pts, max_iter=3000, ftol=1e-14, gtol=1e-10)
        if e_opt < best_energy:
            best_pts = pts_opt
            best_energy = e_opt
    except Exception:
        pass
    
    elapsed = time.time() - start
    if elapsed > time_budget * 0.5:
        # Quick polish if time allows
        if time.time() - start < time_budget * 0.9:
            try:
                cd_pts, cd_e = _high_precision_polish(best_pts, max_rounds=50, time_budget=min(2.0, time_budget - (time.time() - start)))
                if cd_e < best_energy:
                    best_pts = cd_pts
                    best_energy = cd_e
            except Exception:
                pass
        return best_pts, best_energy
    
    # Phase 2: Second L-BFGS with tighter tolerances
    try:
        pts_opt2, e_opt2 = _lbfgs_optimize(best_pts, max_iter=5000, ftol=1e-16, gtol=1e-14)
        if e_opt2 < best_energy:
            best_pts = pts_opt2
            best_energy = e_opt2
    except Exception:
        pass
    
    # Phase 3: High-precision polish
    remaining = time_budget - (time.time() - start)
    if remaining > 0.5:
        try:
            cd_pts, cd_e = _high_precision_polish(best_pts, max_rounds=200, time_budget=min(5.0, remaining))
            if cd_e < best_energy:
                best_pts = cd_pts
                best_energy = cd_e
        except Exception:
            pass
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    configs = []
    
    # Sobol sequences (low discrepancy)
    for s in range(12):
        configs.append(_sobol_sphere(N, seed=s))
    
    # Fibonacci spirals with various offsets
    for offset in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        configs.append(_fibonacci_spiral(N, offset=offset))
    
    # Jittered Fibonacci
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(6):
        jitter_scale = 0.02 + 0.015 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        configs.append(_project(jittered))
    
    # Random configurations
    for s in range(8):
        configs.append(_random_sphere(N, np.random.default_rng(100 + s)))
    
    # Jittered Sobol
    for s in range(6):
        sobol_pts = _sobol_sphere(N, seed=s)
        jitter_scale = 0.02 + 0.01 * s
        jittered = sobol_pts + jitter_scale * rng.standard_normal((N, 3))
        configs.append(_project(jittered))
    
    # Apply repulsive pre-spread to all configs
    spread_configs = []
    for cfg in configs:
        if time.time() > deadline - 30.0:
            break
        spread = _repulsive_prespread(cfg, n_steps=100, strength=1.0, time_budget=0.5)
        spread_configs.append(spread)
    
    # Score initializations
    scored = []
    for cfg in spread_configs:
        if time.time() > deadline - 25.0:
            break
        e_quick = _precise_energy(cfg)
        scored.append((e_quick, cfg))
    
    scored.sort(key=lambda x: x[0])
    
    # Optimize top candidates
    best_pts = None
    best_energy = float('inf')
    top_k = min(25, len(scored))
    
    for i in range(top_k):
        if time.time() > deadline - 15.0:
            break
        
        e_quick, init_pts = scored[i]
        
        # Skip if initial energy is much worse than current best
        if best_pts is not None and e_quick > best_energy + 5.0 and i > 3:
            continue
        
        # Allocate time based on rank
        if i < 3:
            time_budget = 12.0
        elif i < 10:
            time_budget = 8.0
        elif i < 20:
            time_budget = 5.0
        else:
            time_budget = 3.0
        
        pts_opt, e_opt = _local_optimize(init_pts, time_budget=time_budget)
        if e_opt < best_energy:
            best_energy = e_opt
            best_pts = pts_opt.copy()
    
    # Final polish on the global best
    if best_pts is not None and time.time() < deadline - 5.0:
        remaining = deadline - time.time() - 3.0
        if remaining > 8.0:
            best_pts, best_energy = _local_optimize(best_pts, time_budget=remaining)
    
    # Fallback
    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)
    
    # Final normalization and validation
    best_pts = _project(best_pts)
    
    # Check minimum distance
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    min_dist = np.sqrt(np.min(dist_sq))
    if min_dist < 1e-9:
        best_pts = _fibonacci_spiral(N, offset=0.5)
    
    # Ensure finite
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N, offset=0.5)
    
    # Final high-precision polish
    if time.time() < deadline - 1.0:
        try:
            cd_pts, cd_e = _high_precision_polish(best_pts, max_rounds=100, time_budget=min(5.0, deadline - time.time() - 0.5))
            if cd_e < best_energy:
                best_pts = cd_pts
                best_energy = cd_e
        except Exception:
            pass
    
    return best_pts
