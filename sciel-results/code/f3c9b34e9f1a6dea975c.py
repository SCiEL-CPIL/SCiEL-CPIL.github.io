"""Logarithmic energy minimisation for N points on the unit sphere."""
import time
import numpy as np
import math
from scipy.optimize import minimize

N = 150
# Deadline guard: leave a safety margin under the 150 s limit
_DEADLINE = time.time() + 140.0


def _energy_and_grad(pts):
    """Compute energy and tangent-projected gradient efficiently."""
    m = len(pts)
    diff = pts[:, None, :] - pts[None, :, :]        # (m,m,3)
    dist2 = np.einsum('ijk,ijk->ij', diff, diff)    # (m,m)
    np.fill_diagonal(dist2, 1.0)                     # avoid div-by-zero
    dist = np.sqrt(dist2)
    
    # Energy: sum_{i<j} -log(dist_ij)
    # Use upper triangle only
    iu = np.triu_indices(m, k=1)
    d_ij = dist[iu]
    # Guard against zero distances
    d_ij = np.maximum(d_ij, 1e-15)
    E = -np.sum(np.log(d_ij))
    
    # Gradient: grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    inv_d2 = 1.0 / dist2
    grad = np.einsum('ij,ijk->ik', inv_d2, diff)     # (m,3)
    # Project to tangent space
    grad = grad - np.sum(grad * pts, axis=1, keepdims=True) * pts
    return E, grad


def _energy_exact(pts):
    """Exact energy using math.fsum for final scoring."""
    terms = []
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return 1e18
            terms.append(-math.log(d))
    return math.fsum(terms)


def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3))
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _perturb(pts, seed, sigma=0.05):
    rng = np.random.default_rng(seed)
    p = pts + rng.normal(0, sigma, pts.shape)
    p = p / np.linalg.norm(p, axis=1, keepdims=True)
    return p


def _lbfgs_sphere(pts, max_iters=5000):
    """L-BFGS-B optimization on the sphere with tangent-space projection."""
    def obj_and_grad(flat_pts):
        p = flat_pts.reshape(-1, 3)
        p = p / np.linalg.norm(p, axis=1, keepdims=True)
        E, g = _energy_and_grad(p)
        # Flatten gradient
        g_flat = g.flatten()
        return E, g_flat
    
    x0 = pts.flatten()
    res = minimize(
        obj_and_grad, x0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': max_iters, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
    )
    pts_opt = res.x.reshape(-1, 3)
    pts_opt = pts_opt / np.linalg.norm(pts_opt, axis=1, keepdims=True)
    return pts_opt, res


def _polish(pts, max_iters=2000):
    """Final tight polish with very small tolerance."""
    def obj_and_grad(flat_pts):
        p = flat_pts.reshape(-1, 3)
        p = p / np.linalg.norm(p, axis=1, keepdims=True)
        E, g = _energy_and_grad(p)
        g_flat = g.flatten()
        return E, g_flat
    
    x0 = pts.flatten()
    res = minimize(
        obj_and_grad, x0,
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': max_iters, 'ftol': 1e-18, 'gtol': 1e-15, 'maxls': 30}
    )
    pts_opt = res.x.reshape(-1, 3)
    pts_opt = pts_opt / np.linalg.norm(pts_opt, axis=1, keepdims=True)
    return pts_opt, res


def construct():
    t0 = time.time()
    
    # Generate diverse initializations
    inits = []
    
    # 1. Fibonacci sphere (deterministic good start)
    inits.append(_fibonacci_sphere(N))
    
    # 2. Random starts with different seeds
    for seed in [42, 123, 456, 789, 101112]:
        inits.append(_random_sphere(N, seed))
    
    # 3. Perturbed Fibonacci
    for seed in [1, 2, 3, 4, 5]:
        for sigma in [0.05, 0.1, 0.15]:
            inits.append(_perturb(inits[0], seed, sigma))
    
    best_pts = None
    best_E = float('inf')
    
    # Optimize each initialization
    for idx, init in enumerate(inits):
        if time.time() > _DEADLINE - 20:
            break
        
        # First pass: L-BFGS with moderate tolerance
        pts_opt, res = _lbfgs_sphere(init, max_iters=3000)
        
        # Check if we have time for a tighter polish
        if time.time() < _DEADLINE - 10:
            pts_polished, res2 = _polish(pts_opt, max_iters=1000)
            E_polished = _energy_exact(pts_polished)
            if E_polished < best_E:
                best_E = E_polished
                best_pts = pts_polished
        else:
            E_opt = _energy_exact(pts_opt)
            if E_opt < best_E:
                best_E = E_opt
                best_pts = pts_opt
    
    # Final refinement on the best point found
    if best_pts is not None and time.time() < _DEADLINE - 5:
        best_pts, _ = _polish(best_pts, max_iters=500)
    
    # Ensure valid output
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    # Final sanity check
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
