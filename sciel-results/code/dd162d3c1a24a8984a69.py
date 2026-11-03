"""Logarithmic energy minimization for N points on the unit sphere."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100
_DEADLINE = 150.0
_SAFETY = 15.0
_START = time.time()

# Precompute upper triangle indices for N=100
_IU = np.triu_indices(N, k=1)


def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)


def _fast_energy_and_grad(pts):
    """Compute logarithmic energy and gradient using vectorized operations."""
    n = pts.shape[0]
    dot_products = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot_products
    np.fill_diagonal(dist_sq, 1.0)
    np.maximum(dist_sq, 1e-15, out=dist_sq)
    
    energy = -0.5 * np.sum(np.log(dist_sq[_IU]))
    
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    W_row_sum = W.sum(axis=1, keepdims=True)
    grad = W @ pts - W_row_sum * pts
    
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad -= dot * pts
    
    return float(energy), grad


def _fast_energy_only(pts):
    """Compute just the logarithmic energy (faster, no gradient)."""
    dot_products = pts @ pts.T
    dist_sq = 2.0 - 2.0 * dot_products
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-15)
    return float(-0.5 * np.sum(np.log(dist_sq[_IU])))


def _precise_energy(pts):
    """Compute energy using math.fsum for accuracy."""
    n = pts.shape[0]
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-9:
                return float('inf')
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _project(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(seed):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((N, 3))
    return _project(v)

def _fps_init(n_points, n_candidates, seed):
    """Farthest Point Sampling initialization."""
    rng = np.random.default_rng(seed)
    # Generate candidates using Fibonacci spiral for good spread
    candidates = _fibonacci_spiral(n_candidates)
    
    # Shuffle to avoid bias in the first pick
    perm = rng.permutation(n_candidates)
    candidates = candidates[perm]
    
    chosen = [candidates[0]]
    min_dists = np.linalg.norm(candidates - candidates[0], axis=1)
    
    for _ in range(n_points - 1):
        # Pick the candidate with max min_dist to chosen
        idx = np.argmax(min_dists)
        chosen.append(candidates[idx])
        # Update min_dists: min(current_min_dist, dist(new_point, candidate))
        new_dists = np.linalg.norm(candidates - candidates[idx], axis=1)
        min_dists = np.minimum(min_dists, new_dists)
    
    return np.array(chosen)


def _sobol_sphere(seed):
    """Generate points on the sphere using Sobol sequence."""
    sampler = qmc.Sobol(d=3, seed=seed)
    samples = sampler.random(N)
    samples = samples - 0.5
    return _project(samples)


def _valid(X):
    if X.shape != (N, 3) or not np.all(np.isfinite(X)):
        return False
    if np.any(np.linalg.norm(X, axis=1) < 1e-12):
        return False
    d2 = np.einsum('ij,ij->i', X, X)[:, None] + np.einsum('ij,ij->i', X, X)[None, :] \
         - 2.0 * (X @ X.T)
    np.fill_diagonal(d2, np.inf)
    if np.min(d2) < (1e-9) ** 2:
        return False
    return True


def _gradient_flow(pts, rng, n_steps, lr_start, lr_end, deadline):
    """Gradient flow with backtracking line search to explore configuration space."""
    n = pts.shape[0]
    current = pts.copy()
    best = pts.copy()
    best_energy = _fast_energy_only(pts)
    
    for step in range(n_steps):
        if time.time() > deadline:
            break
        
        progress = step / max(n_steps - 1, 1)
        lr_init = lr_start * (lr_end / lr_start) ** progress
        
        e, g = _fast_energy_and_grad(current)
        
        # Add small noise for exploration (annealed)
        noise_scale = 0.005 * (1.0 - progress) ** 1.5
        noise = rng.standard_normal((n, 3)) * noise_scale
        g_noisy = g + noise
        
        # Backtracking line search
        alpha = lr_init
        improved = False
        for _ in range(10):
            candidate = _project(current - alpha * g_noisy)
            e_new = _fast_energy_only(candidate)
            if e_new < e - 1e-10:
                current = candidate
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            # If line search fails, try a pure noise step or break
            # A pure noise step might help escape
            candidate = _project(current + rng.standard_normal((n, 3)) * 0.01)
            e_new = _fast_energy_only(candidate)
            if e_new < best_energy:
                current = candidate
                best_energy = e_new
                best = current.copy()
            # Don't break immediately, allow next step to try again with smaller lr
            continue
        
        e_new = _fast_energy_only(current)
        if e_new < best_energy:
            best_energy = e_new
            best = current.copy()
    
    return best, best_energy


def _optimize_from_start(pts, deadline):
    """Run L-BFGS-B optimization from a starting point, respecting deadline."""
    n = pts.shape[0]
    best_pts = pts.copy()
    best_energy = _precise_energy(pts)
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        np.maximum(norms, 1e-12, out=norms)
        p /= norms
        
        e, g = _fast_energy_and_grad(p)
        return e, g.reshape(-1)
    
    # Phase 1: coarse optimization
    if time.time() < deadline - 3.0:
        try:
            result = minimize(
                objective_and_grad,
                pts.reshape(-1),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 300, 'maxfun': 1500, 'ftol': 1e-10, 'gtol': 1e-6, 'maxls': 20}
            )
            new_pts = _project(result.x.reshape(n, 3))
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_pts = new_pts
                best_energy = e_precise
        except Exception:
            pass
    
    # Phase 2: medium refinement
    if time.time() < deadline - 3.0:
        try:
            result = minimize(
                objective_and_grad,
                best_pts.reshape(-1),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'maxfun': 2500, 'ftol': 1e-13, 'gtol': 1e-9, 'maxls': 30}
            )
            new_pts = _project(result.x.reshape(n, 3))
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_pts = new_pts
                best_energy = e_precise
        except Exception:
            pass
    
    # Phase 3: fine refinement
    if time.time() < deadline - 2.0:
        try:
            result = minimize(
                objective_and_grad,
                best_pts.reshape(-1),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 800, 'maxfun': 4000, 'ftol': 1e-15, 'gtol': 1e-11, 'maxls': 40}
            )
            new_pts = _project(result.x.reshape(n, 3))
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_pts = new_pts
                best_energy = e_precise
        except Exception:
            pass
    
    return best_pts, best_energy


def _final_polish(pts, deadline):
    """Final ultra-precise polish with tiny perturbations."""
    n = pts.shape[0]
    best_pts = pts.copy()
    best_energy = _precise_energy(pts)
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        np.maximum(norms, 1e-12, out=norms)
        p /= norms
        e, g = _fast_energy_and_grad(p)
        return e, g.reshape(-1)
    
    rng = np.random.default_rng(12345)
    perturbations = [0.0] + [2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6]
    
    for scale in perturbations:
        if time.time() > deadline:
            break
        
        if scale == 0.0:
            start = best_pts
        else:
            start = _project(best_pts + rng.standard_normal((n, 3)) * scale)
        
        try:
            result = minimize(
                objective_and_grad,
                start.reshape(-1),
                method='L-BFGS-B',
                jac=True,
                options={
                    'maxiter': 15000,
                    'maxfun': 75000,
                    'ftol': 1e-18,
                    'gtol': 1e-14,
                }
            )
            if result.fun < best_energy:
                new_pts = _project(result.x.reshape(n, 3))
                e_precise = _precise_energy(new_pts)
                if e_precise < best_energy:
                    best_pts = new_pts
                    best_energy = e_precise
        except Exception:
            pass
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # Fibonacci spiral with various offsets
    for offset in [0.5, 0.3, 0.7]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
    
    # Jittered Fibonacci spirals
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(4):
        amp = 0.02 + 0.02 * s
        jittered = base + amp * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
    
    # Random points with different seeds
    for s in range(3):
        initializations.append(_random_sphere(100 + s * 13))
    
    # Sobol sequences - better uniform coverage
    for seed in [1, 2, 3]:
        initializations.append(_sobol_sphere(seed))
        
    # Farthest Point Sampling - globally informed spread
    for s in range(4):
        initializations.append(_fps_init(N, 2000, seed=500 + s * 7))
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase A: Quick screen with gradient flow + short optimization
    for idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 35.0:
            break
        
        remaining = deadline - time.time()
        if remaining < 18.0:
            break
        
        # Gradient flow for exploration
        # Backtracking is slower, so allocate more time or fewer steps
        alloc_flow = min(5.0, remaining - 15.0)
        if alloc_flow < 1.0:
            break
        
        flow_deadline = time.time() + alloc_flow
        try:
            flow_pts, flow_energy = _gradient_flow(
                init_pts, rng, 
                n_steps=30, # Reduced steps due to line search cost
                lr_start=0.05, 
                lr_end=0.001,
                deadline=flow_deadline
            )
            init_pts = flow_pts
        except Exception:
            pass
        
        # Short optimization: just enough to identify the basin
        alloc_opt = min(4.0, remaining - 12.0)
        if alloc_opt < 2.0:
            break
        
        restart_deadline = time.time() + alloc_opt
        
        try:
            def obj(flat):
                p = flat.reshape(N, 3)
                norms = np.linalg.norm(p, axis=1, keepdims=True)
                np.maximum(norms, 1e-12, out=norms)
                p /= norms
                e, g = _fast_energy_and_grad(p)
                return e, g.reshape(-1)
            
            result = minimize(
                obj, init_pts.reshape(-1),
                method='L-BFGS-B', jac=True,
                options={'maxiter': 100, 'maxfun': 500, 'ftol': 1e-8, 'gtol': 1e-4, 'maxls': 10}
            )
            new_pts = _project(result.x.reshape(N, 3))
            e_precise = _precise_energy(new_pts)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = new_pts
        except Exception:
            pass
    
    # Phase B: Deep optimization from the best found so far
    if best_pts is not None and time.time() < deadline - 25.0:
        remaining = deadline - time.time()
        deep_deadline = time.time() + min(remaining - 20.0, 55.0)
        optimized, e_precise = _optimize_from_start(best_pts, deep_deadline)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized
    
    # Phase C: Final high-precision polish with tiny perturbations
    if best_pts is not None and time.time() < deadline - 10.0:
        polish_deadline = time.time() + min(remaining - 8.0, 30.0) if 'remaining' in dir() else deadline - 8.0
        polished, e_precise = _final_polish(best_pts, polish_deadline)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = polished
    
    # Phase D: Riemannian Newton-style local polish (small steps along gradient)
    if best_pts is not None and time.time() < deadline - 4.0:
        pts = best_pts.copy()
        n = pts.shape[0]
        for step_iter in range(200):
            if time.time() > deadline - 3.0:
                break
            e, g = _fast_energy_and_grad(pts)
            lr = 0.01 / (1.0 + 0.05 * step_iter)
            pts_new = pts + lr * g
            pts_new = _project(pts_new)
            e_new = _precise_energy(pts_new)
            if e_new < e:
                pts = pts_new
            else:
                lr *= 0.5
                pts_new = pts + lr * g
                pts_new = _project(pts_new)
                e_new = _precise_energy(pts_new)
                if e_new < e:
                    pts = pts_new
                else:
                    break
        
        if _precise_energy(pts) < best_energy:
            best_energy = _precise_energy(pts)
            best_pts = pts
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)
    
    # Final normalization
    best_pts = _project(best_pts)
    
    # Final validation
    final_energy = _precise_energy(best_pts)
    if not np.isfinite(final_energy):
        best_pts = _fibonacci_spiral(N, offset=0.5)
        best_pts = _project(best_pts)
    
    return best_pts
