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


def _sobol_sphere(n, seed=0):
    """Generate points on sphere using Sobol sequence mapped via spherical coordinates."""
    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    # Need at least n points. Sobol requires power of 2.
    m = int(2 ** np.ceil(np.log2(n)))
    y = sampler.random(m)
    
    # Map to sphere
    # theta = arccos(1 - 2 * y[:, 0])
    # phi = 2 * pi * y[:, 1]
    theta = np.arccos(1 - 2 * y[:, 0])
    phi = 2 * np.pi * y[:, 1]
    
    x = np.sin(theta) * np.cos(phi)
    y_coord = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    pts = np.column_stack([x, y_coord, z])
    return pts[:n]


def _fps_init(n_points, n_candidates, seed):
    """Farthest Point Sampling initialization."""
    rng = np.random.default_rng(seed)
    # Generate candidates using Sobol for good spread
    candidates = _sobol_sphere(n_candidates, seed=seed)
    
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


def _optimize(pts, deadline, maxiter, ftol, gtol):
    """Run L-BFGS-B optimization from a starting point."""
    n = pts.shape[0]
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        np.maximum(norms, 1e-12, out=norms)
        p /= norms
        
        e, g = _fast_energy_and_grad(p)
        return e, g.reshape(-1)
    
    try:
        result = minimize(
            objective_and_grad,
            pts.reshape(-1),
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'maxfun': maxiter*10, 'ftol': ftol, 'gtol': gtol, 'maxls': 20}
        )
        new_pts = _project(result.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci spirals with various offsets
    for offset in [0.5, 0.3, 0.7, 0.1, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
        
    # 2. Sobol sequences (Low discrepancy, mapped to sphere)
    for s in range(10):
        initializations.append(_sobol_sphere(N, seed=s))
        
    # 3. Random points with different seeds
    for s in range(10):
        initializations.append(_random_sphere(100 + s * 13))
        
    # 4. Farthest Point Sampling - globally informed spread
    for s in range(5):
        initializations.append(_fps_init(N, 2000, seed=500 + s * 7))
        
    # 5. Jittered Fibonacci spirals
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(5):
        amp = 0.02 + 0.02 * s
        jittered = base + amp * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
        
    # Score all initializations with fast energy
    scored_inits = []
    for init_pts in initializations:
        if time.time() > deadline - 30.0:
            break
        e_quick = _fast_energy_only(init_pts)
        scored_inits.append((e_quick, init_pts))
        
    scored_inits.sort(key=lambda x: x[0])
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Optimize top candidates
    # We have ~115s deadline. 30s for init/screening. ~85s left.
    # Let's take top 20 candidates.
    top_candidates = scored_inits[:20]
    
    for i, (e_quick, init_pts) in enumerate(top_candidates):
        if time.time() > deadline - 20.0:
            break
        
        # Allocate time based on remaining budget
        # First few candidates get more time
        if i < 5:
            maxiter = 500
            ftol = 1e-10
            gtol = 1e-6
        else:
            maxiter = 300
            ftol = 1e-8
            gtol = 1e-5
            
        optimized = _optimize(init_pts, deadline, maxiter, ftol, gtol)
        e_precise = _precise_energy(optimized)
        
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized.copy()
            
    # Phase 2: Deep optimization from the best found so far
    if best_pts is not None and time.time() < deadline - 15.0:
        # Use tighter tolerances and more iterations
        maxiter = 1000
        ftol = 1e-12
        gtol = 1e-8
        
        optimized = _optimize(best_pts, deadline, maxiter, ftol, gtol)
        e_precise = _precise_energy(optimized)
        
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized.copy()
            
    # Phase 3: Final high-precision polish
    if best_pts is not None and time.time() < deadline - 5.0:
        # Very tight tolerances
        maxiter = 2000
        ftol = 1e-14
        gtol = 1e-10
        
        optimized = _optimize(best_pts, deadline, maxiter, ftol, gtol)
        e_precise = _precise_energy(optimized)
        
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized.copy()
            
    # Final small step polish with gradient descent if time permits
    if best_pts is not None and time.time() < deadline - 2.0:
        pts = best_pts.copy()
        n = pts.shape[0]
        for step_iter in range(50):
            if time.time() > deadline - 1.0:
                break
            e, g = _fast_energy_and_grad(pts)
            lr = 0.01 / (1.0 + 0.05 * step_iter)
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
