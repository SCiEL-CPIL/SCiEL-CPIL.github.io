"""Logarithmic energy minimisation for N points on the unit sphere."""
import time
import numpy as np
import math
from scipy.optimize import minimize

N = 150
_DEADLINE = time.time() + 145.0

def _energy(pts):
    """Vectorized sum_{i<j} log(1 / |x_i - x_j|) using math.fsum."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    d = d[d > 1e-15]
    if len(d) == 0:
        return 1e18
    terms = -np.log(d)
    return math.fsum(terms)

def _grad(pts):
    """Vectorized gradient of E wrt each point (tangent-projected)."""
    diff = pts[:, None, :] - pts[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist2, 1.0)
    inv_d2 = 1.0 / dist2
    np.fill_diagonal(inv_d2, 0.0)
    grad = np.einsum('ij,ijk->ik', inv_d2, diff)
    radial_comp = np.sum(grad * pts, axis=1, keepdims=True)
    grad = grad - radial_comp * pts
    return grad

def _retract(pts):
    """Project points back to the sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed):
    rng = np.random.default_rng(seed)
    pts = rng.normal(0, 1, (n, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _lbfgs_optimize(pts, max_time=25.0, tol=1e-12):
    """Optimize using L-BFGS-B with wall-clock time limit."""
    t_start = time.time()
    pts = pts.copy()
    E = _energy(pts)
    best_pts = pts.copy()
    best_E = E
    
    def objective_and_grad(x_flat):
        x = x_flat.reshape(N, 3)
        x = _retract(x)
        e = _energy(x)
        g = _grad(x)
        return e, g.flatten()
    
    x0 = pts.flatten()
    
    result = minimize(
        lambda x: objective_and_grad(x)[0],
        x0,
        jac=lambda x: objective_and_grad(x)[1],
        method='L-BFGS-B',
        options={
            'maxiter': 2000,
            'ftol': tol,
            'gtol': tol,
            'maxfun': 2000,
            'maxls': 50,
            'maxcor': 20
        }
    )
    
    if result.fun < best_E:
        x_opt = _retract(result.x.reshape(N, 3))
        E_opt = _energy(x_opt)
        if E_opt < best_E:
            best_pts = x_opt
            best_E = E_opt
    
    return best_pts, best_E

def _local_search(pts, max_time=10.0):
    """Simple local search with random perturbations."""
    t_start = time.time()
    pts = pts.copy()
    E = _energy(pts)
    best_pts = pts.copy()
    best_E = E
    rng = np.random.default_rng(42)
    
    step_size = 0.05
    while time.time() - t_start < max_time:
        for _ in range(10):
            idx = rng.integers(0, N)
            perturbation = rng.normal(0, step_size, 3)
            new_pts = pts.copy()
            new_pts[idx] = _retract(new_pts[idx] + perturbation)
            new_E = _energy(new_pts)
            if new_E < best_E:
                best_pts = new_pts
                best_E = new_E
                pts = new_pts
                step_size = max(0.01, step_size * 0.95)
            else:
                step_size = min(0.1, step_size * 1.05)
    
    return best_pts, best_E

def construct():
    t0 = time.time()
    
    # 1. Start with Fibonacci spiral
    pts0 = _fibonacci_sphere(N)
    best_pts, best_E = _lbfgs_optimize(pts0, max_time=20.0)
    
    # 2. Multiple diverse restarts
    seeds = [42, 123, 999, 2023, 777, 555, 314, 271, 161, 83, 3141, 2718, 1618, 8321, 4590]
    for seed in seeds:
        if time.time() > _DEADLINE - 40:
            break
        p = _random_sphere(N, seed)
        p2, E2 = _lbfgs_optimize(p, max_time=12.0)
        if E2 < best_E:
            best_pts, best_E = p2, E2
    
    # 3. Jittered Fibonacci for more diversity
    if time.time() < _DEADLINE - 35:
        rng = np.random.default_rng(12345)
        for _ in range(5):
            if time.time() > _DEADLINE - 30:
                break
            p = _fibonacci_sphere(N)
            p += rng.normal(0, 0.15, (N, 3))
            p = _retract(p)
            p2, E2 = _lbfgs_optimize(p, max_time=10.0)
            if E2 < best_E:
                best_pts, best_E = p2, E2
    
    # 4. Local search refinement
    if time.time() < _DEADLINE - 15:
        ls_pts, ls_E = _local_search(best_pts, max_time=8.0)
        if ls_E < best_E:
            best_pts, best_E = ls_pts, ls_E
    
    # 5. Final high-precision polish
    if time.time() < _DEADLINE - 5:
        polished_pts, polished_E = _lbfgs_optimize(best_pts, max_time=8.0, tol=1e-14)
        if polished_E < best_E:
            best_pts, best_E = polished_pts, polished_E

    # Final normalization
    best_pts = _retract(best_pts)
    
    return best_pts
