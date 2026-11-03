"""Logarithmic energy minimization for N points on the unit sphere.
Combines multi-start BFGS optimization with Sobol/Fibonacci initializations,
gradient-flow annealing for broader exploration, and ultra-precise final polish."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100
_DEADLINE = 145.0
_START = time.time()


def _remaining():
    return _DEADLINE - (time.time() - _START)


def _project(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return X / n


def _energy(pts):
    """Fast energy computation using matrix operations."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(dist[iu])))


def _energy_grad(pts):
    """Compute energy and gradient (projected onto tangent space)."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    
    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))
    
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    sum_W = W.sum(axis=1, keepdims=True)
    Wx = W @ pts
    raw_grad = sum_W * pts - Wx
    grad = -raw_grad
    
    dot = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - dot * pts
    
    return energy, grad


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


def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(rng):
    v = rng.standard_normal((N, 3))
    return _project(v)


def _sobol_sphere(n, seed=0):
    """Generate points on sphere using Sobol sequence."""
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


def _optimize(pts, deadline, max_iter=500, ftol=1e-12, gtol=1e-10, method='L-BFGS-B'):
    """Run L-BFGS-B optimization from a starting point with projection."""
    n = pts.shape[0]
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        norms = np.linalg.norm(p, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        p = p / norms
        e, g = _energy_grad(p)
        return e, g.reshape(-1)
    
    try:
        x0 = pts.reshape(-1)
        result = minimize(
            objective_and_grad,
            x0,
            method=method,
            jac=True,
            options={
                'maxiter': max_iter,
                'gtol': gtol,
                'ftol': ftol,
                'maxfun': max_iter * 10,
            }
        )
        new_pts = _project(result.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts


def _anneal(pts, deadline, n_steps=200, base_lr=0.01, min_lr=0.001):
    """Gradient-flow annealing with adaptive learning rate and noise schedule."""
    current = pts.copy()
    n = current.shape[0]
    best_e = _energy(current)
    best_current = current.copy()
    
    for step in range(n_steps):
        if time.time() > deadline - 5.0:
            break
        
        # Cosine annealing schedule with restarts
        progress = step / max(n_steps - 1, 1)
        lr = base_lr * 0.5 * (1 + np.cos(np.pi * progress))
        lr = max(lr, min_lr)
        
        e, g = _energy_grad(current)
        
        # Adaptive noise: larger early, smaller late
        noise_scale = 0.002 * (1 - progress) + 0.0001
        perturbation = noise_scale * np.random.standard_normal((n, 3))
        step_vec = -lr * (g + perturbation)
        
        # Project step to tangent space
        dot = np.sum(step_vec * current, axis=1, keepdims=True)
        step_vec = step_vec - dot * current
        
        current = current + step_vec
        current = _project(current)
        
        # Track best
        e_new = _energy(current)
        if e_new < best_e:
            best_e = e_new
            best_current = current.copy()
    
    return best_current


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


def _newton_polish(pts, deadline, max_iter=50, tol=1e-14):
    """Newton-like local refinement using Hessian approximation."""
    current = pts.copy()
    n = current.shape[0]
    
    for it in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        e, g = _energy_grad(current)
        g_norm = np.linalg.norm(g)
        if g_norm < tol:
            break
        
        # Approximate Hessian via finite differences in tangent space
        # Use a simpler momentum-based approach instead
        lr = 0.1 / (1 + it * 0.1)
        step_vec = -lr * g
        dot = np.sum(step_vec * current, axis=1, keepdims=True)
        step_vec = step_vec - dot * current
        
        current = current + step_vec
        current = _project(current)
    
    return current


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Sobol sequences (Low discrepancy)
    for s in range(15):
        initializations.append(_sobol_sphere(N, seed=s))
        
    # 2. Fibonacci spirals with various offsets
    for offset in [0.1, 0.3, 0.5, 0.7, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
        
    # 3. Jittered Fibonacci spirals
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(8):
        jitter_scale = 0.05 + 0.01 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
        
    # 4. Random points with different seeds
    for s in range(15):
        initializations.append(_random_sphere(np.random.default_rng(100 + s)))
        
    best_pts = None
    best_energy = float('inf')
    
    # Score all initializations
    scored_inits = []
    for init_pts in initializations:
        if time.time() > deadline - 25.0:
            break
        e_quick = _energy(init_pts)
        scored_inits.append((e_quick, init_pts))
        
    scored_inits.sort(key=lambda x: x[0])
    
    # Phase 1: Optimize top candidates with BFGS
    top_candidates = scored_inits[:30]
    
    for e_quick, init_pts in top_candidates:
        if time.time() > deadline - 15.0:
            break
        
        optimized = _optimize(init_pts, deadline, max_iter=800, ftol=1e-14, gtol=1e-11)
        
        e_precise = _precise_energy(optimized)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized.copy()
            
    # Phase 2: Annealing + optimization on the best configuration
    if best_pts is not None and time.time() < deadline - 10.0:
        annealed = _anneal(best_pts, deadline, n_steps=200, base_lr=0.01, min_lr=0.0003)
        refined = _optimize(annealed, deadline, max_iter=1500, ftol=1e-14, gtol=1e-12)
        
        e_precise = _precise_energy(refined)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = refined.copy()
    
    # Phase 3: Final ultra-precise polish with multiple strategies
    if best_pts is not None and time.time() < deadline - 3.0:
        # Try multiple tiny perturbations around the best
        for attempt in range(5):
            if time.time() > deadline - 2.0:
                break
            perturbed = best_pts + 0.0003 * rng.standard_normal((N, 3))
            perturbed = _project(perturbed)
            polished = _optimize(perturbed, deadline, max_iter=2000, ftol=1e-16, gtol=1e-14)
            
            e_precise = _precise_energy(polished)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = polished.copy()
        
        # Newton-like polish
        if time.time() < deadline - 1.5:
            polished = _newton_polish(best_pts, deadline, max_iter=30, tol=1e-15)
            e_precise = _precise_energy(polished)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = polished.copy()
    
    # Final polish on the global best
    if best_pts is not None and time.time() < deadline - 1.0:
        refined = _optimize(best_pts, deadline, max_iter=3000, ftol=1e-16, gtol=1e-15)
        e_precise = _precise_energy(refined)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = refined
    
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
