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

def _halton_sphere(n, base_seed=0):
    """Generate points on sphere using Halton sequence."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    indices = np.arange(1, n + 1)
    u1 = np.array([halton(i, 2) for i in indices])
    u2 = np.array([halton(i, 3) for i in indices])
    
    theta = np.arccos(1 - 2 * u1)
    phi = 2 * np.pi * u2
    
    x = np.sin(theta) * np.cos(phi)
    y_coord = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    pts = np.column_stack([x, y_coord, z])
    return pts


def _optimize(pts, deadline, max_iter=500, ftol=1e-12, gtol=1e-10):
    """Run BFGS optimization from a starting point."""
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
            method='L-BFGS-B',
            jac=True,
            options={
                'maxiter': max_iter,
                'gtol': gtol,
                'ftol': ftol,
                'maxfun': max_iter * 3,
            }
        )
        new_pts = _project(result.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts


def _anneal(pts, deadline, n_steps=200, base_lr=0.01, min_lr=0.001):
    """Gradient-flow annealing with momentum to explore configuration space more broadly."""
    current = pts.copy()
    n = current.shape[0]
    velocity = np.zeros_like(current)
    momentum = 0.9
    
    for step in range(n_steps):
        if time.time() > deadline - 5.0:
            break
        
        lr = base_lr * (min_lr / base_lr) ** (step / max(n_steps - 1, 1))
        
        e, g = _energy_grad(current)
        # Add small random perturbation to escape local minima
        perturbation = 0.0005 * np.random.standard_normal((n, 3))
        grad_with_noise = g + perturbation
        
        # Update velocity with momentum
        velocity = momentum * velocity - lr * grad_with_noise
        
        # Project step to tangent space
        dot = np.sum(velocity * current, axis=1, keepdims=True)
        velocity = velocity - dot * current
        
        current = current + velocity
        current = _project(current)
    
    return current


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
        jitter_scale = 0.03 + 0.008 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
        
    # 4. Random points with different seeds
    for s in range(15):
        initializations.append(_random_sphere(np.random.default_rng(100 + s)))
    
    # 5. Halton sequences
    for s in range(5):
        initializations.append(_halton_sphere(N))
        
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
    
    # Phase 1: Optimize top candidates with L-BFGS-B
    top_candidates = scored_inits[:30]
    
    for e_quick, init_pts in top_candidates:
        if time.time() > deadline - 15.0:
            break
        
        optimized = _optimize(init_pts, deadline, max_iter=1000, ftol=1e-14, gtol=1e-12)
        
        e_precise = _precise_energy(optimized)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = optimized.copy()
            
    # Phase 2: Annealing + optimization on the best configuration
    if best_pts is not None and time.time() < deadline - 10.0:
        annealed = _anneal(best_pts, deadline, n_steps=200, base_lr=0.006, min_lr=0.0003)
        refined = _optimize(annealed, deadline, max_iter=1500, ftol=1e-14, gtol=1e-12)
        
        e_precise = _precise_energy(refined)
        if e_precise < best_energy:
            best_energy = e_precise
            best_pts = refined.copy()
    
    # Phase 3: Final ultra-precise polish with multiple restarts
    if best_pts is not None and time.time() < deadline - 5.0:
        # Try multiple tiny perturbations around the best
        for attempt in range(5):
            if time.time() > deadline - 3.0:
                break
            perturbation_scale = 0.0003 + 0.0001 * attempt
            perturbed = best_pts + perturbation_scale * rng.standard_normal((N, 3))
            perturbed = _project(perturbed)
            polished = _optimize(perturbed, deadline, max_iter=2000, ftol=1e-15, gtol=1e-13)
            
            e_precise = _precise_energy(polished)
            if e_precise < best_energy:
                best_energy = e_precise
                best_pts = polished.copy()
    
    # Final polish on the global best
    if best_pts is not None and time.time() < deadline - 1.0:
        refined = _optimize(best_pts, deadline, max_iter=3000, ftol=1e-16, gtol=1e-14)
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
