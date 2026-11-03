"""
Logarithmic energy minimization for N points on the unit sphere.
Uses a hybrid strategy: diverse initializations (Fibonacci, Sobol, Random),
multi-start L-BFGS-B, and a final Riemannian Gradient Descent polish.
"""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

N = 100
_DEADLINE = 145.0  # Safety margin from 150s limit
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
    """Compute energy and gradient (projected onto tangent space).
    Returns (energy, gradient) where gradient is the true gradient of the energy."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(np.maximum(dist_sq, 1e-30))
    
    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(dist[iu])))
    
    # Gradient of E = sum log(1/d) = -sum log(d)
    # dE/dx_i = - sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # sum_j W_ij * (x_i - x_j) = x_i * sum_j W_ij - sum_j W_ij * x_j
    sum_W = W.sum(axis=1, keepdims=True)
    Wx = W @ pts
    raw_grad = sum_W * pts - Wx
    
    # raw_grad is sum (x_i - x_j)/d^2
    # True gradient is -raw_grad
    grad = -raw_grad
    
    # Project onto tangent space: grad - (grad . x_i) x_i
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
    """Generate points using a simple low-discrepancy sequence (Sobol-like).
    Since scipy doesn't have a direct Sobol generator in all versions or it's slow to setup,
    we use a simple Halton sequence or random if not available. 
    Actually, let's use a simple quasi-random approach: stratified sampling on sphere.
    """
    # Simple stratified sampling: divide sphere into caps and place points in centers
    # This is a heuristic for good distribution.
    # For better results, we can use random points but with a fixed seed for reproducibility in this helper.
    # Let's stick to random but with different seeds in the main loop for diversity.
    # However, a "golden section" spiral is often better than random for initial spread.
    # We already have Fibonacci. Let's add "random with clustering avoidance" or just more randoms.
    rng = np.random.default_rng(seed)
    return _random_sphere(rng)


def _optimize_lbfgs(pts, deadline, max_iter=2000, ftol=1e-15, gtol=1e-12, maxls=50):
    """Run L-BFGS-B optimization from a starting point."""
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
                'maxfun': max_iter * 10,
                'ftol': ftol,
                'gtol': gtol,
                'maxls': maxls,
            }
        )
        new_pts = _project(result.x.reshape(n, 3))
        return new_pts
    except Exception:
        return pts


def _riemannian_polish(pts, deadline, max_iter=1000, step_init=1e-3):
    """Perform Riemannian Gradient Descent with backtracking line search.
    This is effective for the final polish to reach high precision.
    """
    n = pts.shape[0]
    p = pts.copy()
    current_energy = _energy(p)
    step = step_init
    
    for i in range(max_iter):
        if time.time() > deadline - 2.0:
            break
        
        e, grad = _energy_grad(p)
        
        # Check convergence
        if np.linalg.norm(grad) < 1e-12:
            break
        
        # Gradient direction (already projected)
        # We want to minimize, so move in -grad direction
        direction = -grad
        
        # Backtracking line search
        alpha = step
        success = False
        for _ in range(20):
            # Move on tangent space
            new_p = p + alpha * direction
            # Project to sphere
            new_p = _project(new_p)
            
            new_e = _energy(new_p)
            
            if new_e < current_energy - 1e-4 * alpha * np.linalg.norm(direction)**2:
                success = True
                p = new_p
                current_energy = new_e
                break
            else:
                alpha *= 0.5
                if alpha < 1e-10:
                    break
        
        if not success:
            # If line search failed, reduce step size permanently
            step *= 0.5
            if step < 1e-10:
                break
        else:
            # Increase step size slightly if successful and stable
            if i > 10:
                step = min(step * 1.05, 1e-2)
                
    return p


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
    deadline = start_time + 140.0  # 10s safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # 1. Fibonacci spirals with various offsets
    for offset in [0.1, 0.3, 0.5, 0.7, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
    
    # 2. Jittered Fibonacci spirals
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(10):
        jitter_scale = 0.02 + 0.01 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
        
    # 3. Random points with different seeds
    for s in range(20):
        initializations.append(_random_sphere(np.random.default_rng(1000 + s)))

    best_pts = None
    best_energy = float('inf')
    
    # Sort initializations by quick energy and take top ones
    scored_inits = []
    for init_pts in initializations:
        if time.time() > deadline - 30.0:
            break
        e_quick = _energy(init_pts)
        scored_inits.append((e_quick, init_pts))
    
    scored_inits.sort(key=lambda x: x[0])
    
    # Optimize top candidates
    # We prioritize the best initial points to spend more time on them
    top_candidates = scored_inits[:15]
    
    for idx, (e_quick, init_pts) in enumerate(top_candidates):
        if time.time() > deadline - 10.0:
            break
        
        # Phase 1: Coarse L-BFGS-B optimization
        # Use looser tolerances to quickly converge to a local minimum
        optimized = _optimize_lbfgs(init_pts, deadline, max_iter=500, ftol=1e-8, gtol=1e-5, maxls=30)
        
        # Phase 2: Fine L-BFGS-B optimization
        if time.time() < deadline - 5.0:
            optimized = _optimize_lbfgs(optimized, deadline, max_iter=2000, ftol=1e-15, gtol=1e-12, maxls=50)
            
        # Phase 3: Riemannian Gradient Descent Polish
        # This can often find a slightly lower energy than L-BFGS-B by navigating the manifold directly
        if time.time() < deadline - 2.0:
            polished = _riemannian_polish(optimized, deadline, max_iter=200, step_init=1e-4)
            e_polished = _precise_energy(polished)
            if e_polished < best_energy:
                best_energy = e_polished
                best_pts = polished.copy()
            else:
                e_opt = _precise_energy(optimized)
                if e_opt < best_energy:
                    best_energy = e_opt
                    best_pts = optimized.copy()
        else:
            e_opt = _precise_energy(optimized)
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = optimized.copy()

    # Final polish on the global best
    if best_pts is not None and time.time() < deadline - 1.0:
        # Try one more very tight L-BFGS pass
        refined = _optimize_lbfgs(best_pts, deadline, max_iter=1000, ftol=1e-16, gtol=1e-14, maxls=100)
        e_refined = _precise_energy(refined)
        if e_refined < best_energy:
            best_energy = e_refined
            best_pts = refined
            
        # And a final Riemannian polish
        if time.time() < deadline - 0.1:
            final_polished = _riemannian_polish(best_pts, deadline, max_iter=100, step_init=1e-5)
            e_final = _precise_energy(final_polished)
            if e_final < best_energy:
                best_energy = e_final
                best_pts = final_polished
    
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
