"""Logarithmic energy minimization for N points on the unit sphere.
Combines Riemannian gradient descent with diverse low-discrepancy initializations
(Sobol, Fibonacci, random) and multi-stage BFGS polishing."""
import time
import math
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

N = 100
_DEADLINE = 150.0
_SAFETY = 10.0
_START = time.time()


def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START)


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
    dist_sq = np.maximum(dist_sq, 1e-30)
    iu = np.triu_indices(n, k=1)
    return float(-np.sum(np.log(np.sqrt(dist_sq[iu]))))


def _energy_grad(pts):
    """Compute energy and gradient (projected onto tangent space)."""
    n = pts.shape[0]
    sq_norms = np.einsum('ij,ij->i', pts, pts)
    dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2.0 * (pts @ pts.T)
    np.fill_diagonal(dist_sq, 1.0)
    dist_sq = np.maximum(dist_sq, 1e-30)
    
    iu = np.triu_indices(n, k=1)
    energy = float(-np.sum(np.log(np.sqrt(dist_sq[iu]))))
    
    # Gradient of -sum log(d_ij) with respect to points
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    row_sums = W.sum(axis=1, keepdims=True)
    grad = -row_sums * pts + (W @ pts)
    
    # Project onto tangent space
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
    """Generate well-distributed points on sphere using Sobol sequence."""
    sampler = qmc.Sobol(d=2, scramble=True, seed=seed)
    m = int(2 ** np.ceil(np.log2(n)))
    y = sampler.random(m)
    
    theta = np.arccos(1 - 2 * y[:, 0])
    phi = 2 * np.pi * y[:, 1]
    
    x = np.sin(theta) * np.cos(phi)
    y_coord = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    
    pts = np.column_stack([x, y_coord, z])
    return pts[:n]


def _exponential_map(points, tangent_vecs):
    """Riemannian exponential map on the sphere."""
    norms = np.linalg.norm(tangent_vecs, axis=1, keepdims=True)
    norms_safe = np.where(norms < 1e-12, 1.0, norms)
    unit_tangent = tangent_vecs / norms_safe
    cos_t = np.cos(norms)
    sin_t = np.sin(norms)
    new_points = points * cos_t + unit_tangent * sin_t
    return _project(new_points)


def _gradient_descent(pts, deadline, max_iter=5000, lr_init=0.01, lr_min=1e-14):
    """Adaptive Riemannian gradient descent with backtracking line search."""
    current = pts.copy()
    best_e = _energy(current)
    best_pts = current.copy()
    lr = lr_init
    c1 = 0.1
    rho = 0.5
    no_improve_count = 0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        e, g = _energy_grad(current)
        g_norm = np.linalg.norm(g)
        if g_norm < 1e-13:
            break
        
        step = lr
        found = False
        for bt in range(20):
            v = -step * g
            new_pts = _exponential_map(current, v)
            new_e = _energy(new_pts)
            
            if new_e <= e + c1 * step * g_norm * g_norm:
                found = True
                break
            step *= rho
        
        if not found:
            step = max(lr * rho**20, 1e-15)
            v = -step * g
            new_pts = _exponential_map(current, v)
            new_e = _energy(new_pts)
        
        if new_e < best_e - 1e-15:
            current = new_pts
            best_e = new_e
            best_pts = current.copy()
            no_improve_count = 0
            lr = min(step * 1.5, 0.1)
        elif new_e < best_e:
            current = new_pts
            no_improve_count += 1
            lr = step
        else:
            lr = step * 0.5
            no_improve_count += 1
            if lr < lr_min or no_improve_count > 50:
                break
    
    return best_pts, best_e


def _momentum_descent(pts, deadline, max_iter=3000, lr_init=0.03, beta=0.9):
    """Momentum-based Riemannian gradient descent."""
    current = pts.copy()
    best_e = _energy(current)
    best_pts = current.copy()
    
    velocity = np.zeros_like(current)
    lr = lr_init
    c1 = 0.1
    rho = 0.5
    no_improve_count = 0
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        e, g = _energy_grad(current)
        g_norm = np.linalg.norm(g)
        
        if g_norm < 1e-12:
            break
        
        # Project velocity to tangent space
        dot_v = np.sum(velocity * current, axis=1, keepdims=True)
        velocity = velocity - dot_v * current
        
        step = lr
        found = False
        for bt in range(15):
            v = velocity - step * g
            new_pts = _exponential_map(current, v)
            new_e = _energy(new_pts)
            
            if new_e <= e + c1 * step * g_norm * g_norm:
                found = True
                break
            step *= rho
        
        if not found:
            step = max(lr * rho**15, 1e-15)
            v = velocity - step * g
            new_pts = _exponential_map(current, v)
            new_e = _energy(new_pts)
        
        if new_e < best_e - 1e-15:
            best_e = new_e
            best_pts = new_pts.copy()
            current = new_pts
            velocity = beta * velocity - step * g
            no_improve_count = 0
            lr = min(step * 1.5, 0.1)
        elif new_e < best_e:
            current = new_pts
            velocity = beta * velocity - step * g
            no_improve_count += 1
            lr = step
        else:
            current = best_pts.copy()
            velocity *= 0.5
            lr = step * 0.5
            no_improve_count += 1
            if no_improve_count > 50:
                break
    
    return best_pts, best_e


def _bfgs_polish(pts, deadline, max_iter=5000, ftol=1e-18, gtol=1e-14):
    """High-precision BFGS polish on flattened coordinates."""
    n = pts.shape[0]
    
    def objective_and_grad(flat):
        p = flat.reshape(n, 3)
        p = _project(p)
        e, g = _energy_grad(p)
        return e, g.flatten()
    
    try:
        x0 = pts.flatten()
        result = minimize(objective_and_grad, x0, method='BFGS', jac=True,
                         options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol})
        best_pts = _project(result.x.reshape(n, 3))
        best_e = _energy(best_pts)
        return best_pts, best_e
    except Exception:
        return pts, _energy(pts)


def _lbfgsb_polish(pts, deadline, max_iter=10000, ftol=1e-18, gtol=1e-14):
    """L-BFGS-B polish with very tight tolerances."""
    n = pts.shape[0]
    
    def objective(flat):
        p = flat.reshape(n, 3)
        p = _project(p)
        return _energy(p)
    
    def gradient(flat):
        p = flat.reshape(n, 3)
        p = _project(p)
        _, g = _energy_grad(p)
        return g.flatten()
    
    try:
        x0 = pts.flatten()
        result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                         options={'maxiter': max_iter, 'ftol': ftol, 'gtol': gtol,
                                  'maxls': 100, 'maxfun': 50000})
        best_pts = _project(result.x.reshape(n, 3))
        best_e = _energy(best_pts)
        return best_pts, best_e
    except Exception:
        return pts, _energy(pts)


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
    deadline = start_time + 145.0  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    initializations = []
    
    # Sobol sequences (best low-discrepancy)
    for s in range(5):
        initializations.append(_sobol_sphere(N, seed=s))
    
    # Fibonacci spirals with various offsets
    for offset in [0.1, 0.3, 0.5, 0.7, 0.9]:
        initializations.append(_fibonacci_spiral(N, offset=offset))
    
    # Jittered Fibonacci spirals
    base = _fibonacci_spiral(N, offset=0.5)
    for s in range(5):
        jitter_scale = 0.01 + 0.005 * s
        jittered = base + jitter_scale * rng.standard_normal((N, 3))
        initializations.append(_project(jittered))
    
    best_pts = None
    best_energy = float('inf')
    
    # Quick evaluation and ranking
    scored_inits = []
    for init_pts in initializations:
        if time.time() > deadline - 30.0:
            break
        e_quick = _energy(init_pts)
        scored_inits.append((e_quick, init_pts))
    
    scored_inits.sort(key=lambda x: x[0])
    
    # Optimize top 5 candidates with multiple methods
    top_candidates = scored_inits[:5]
    for e_quick, init_pts in top_candidates:
        if time.time() > deadline - 40.0:
            break
        
        # Riemannian gradient descent
        gd_pts, gd_e = _gradient_descent(init_pts, deadline, max_iter=2000, lr_init=0.01)
        if gd_e < best_energy:
            best_energy = gd_e
            best_pts = gd_pts
        
        # Momentum descent
        md_pts, md_e = _momentum_descent(init_pts, deadline, max_iter=1500, lr_init=0.03)
        if md_e < best_energy:
            best_energy = md_e
            best_pts = md_pts
    
    # Multi-stage polishing on the global best
    if best_pts is not None:
        # Stage 1: BFGS polish (robust)
        if time.time() < deadline - 20.0:
            refined, ref_e = _bfgs_polish(best_pts, deadline, max_iter=10000)
            if ref_e < best_energy:
                best_energy = ref_e
                best_pts = refined
        
        # Stage 2: L-BFGS-B with very tight tolerances
        if time.time() < deadline - 10.0:
            refined, ref_e = _lbfgsb_polish(best_pts, deadline, max_iter=20000, ftol=1e-20, gtol=1e-16)
            if ref_e < best_energy:
                best_energy = ref_e
                best_pts = refined
        
        # Stage 3: Final ultra-precise gradient descent to settle
        if time.time() < deadline - 2.0:
            refined, ref_e = _gradient_descent(best_pts, deadline, max_iter=5000, lr_init=0.0001, lr_min=1e-18)
            if ref_e < best_energy:
                best_energy = ref_e
                best_pts = refined
        
        # Stage 4: One last L-BFGS-B pass with even tighter settings
        if time.time() < deadline - 1.0:
            refined, ref_e = _lbfgsb_polish(best_pts, deadline, max_iter=5000, ftol=1e-25, gtol=1e-20)
            if ref_e < best_energy:
                best_energy = ref_e
                best_pts = refined
    
    if best_pts is None:
        best_pts = _fibonacci_spiral(N, offset=0.5)
    
    # Final normalization and validation
    best_pts = _project(best_pts)
    
    final_energy = _precise_energy(best_pts)
    if not np.isfinite(final_energy):
        best_pts = _fibonacci_spiral(N, offset=0.5)
        best_pts = _project(best_pts)
    
    return best_pts
