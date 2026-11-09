"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
from scipy.optimize import minimize


def _energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    i_idx, j_idx = np.triu_indices(n, k=1)
    dists = np.sqrt(dist_sq[i_idx, j_idx])
    dists = np.maximum(dists, 1e-300)
    return -np.sum(np.log(dists))


def _energy_and_grad(pts):
    """Compute logarithmic energy and its gradient (tangent to sphere)."""
    n = pts.shape[0]
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    
    # Gradient of -log(d) wrt x_k = (x_k - x_j) / |x_k - x_j|^2 summed over j!=k
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad_raw = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    # Project gradient to tangent space: g_tan = g - (g . x) * x
    dot = np.sum(grad_raw * pts, axis=1, keepdims=True)
    grad = grad_raw - dot * pts
    
    # Energy
    i_idx, j_idx = np.triu_indices(n, k=1)
    dists = np.sqrt(dist_sq[i_idx, j_idx])
    dists = np.maximum(dists, 1e-300)
    E = -np.sum(np.log(dists))
    
    return E, grad.flatten()


def _flatten(pts):
    return pts.flatten()


def _unflatten(flat):
    return flat.reshape(N, 3)


def _objective(flat):
    pts = _unflatten(flat)
    # Normalize each point to unit sphere
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    E, _ = _energy_and_grad(pts)
    return E


def _grad(flat):
    pts = _unflatten(flat)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    pts = pts / norms
    _, g = _energy_and_grad(pts)
    return g


def _fibonacci_sphere(n, seed=0):
    """Generate Fibonacci spiral points on the sphere with optional jitter."""
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if seed > 0:
        # Add small random perturbation
        pts += rng.randn(n, 3) * 0.01
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
    return pts


def construct():
    best_pts = None
    best_E = np.inf
    
    # Multiple restarts with different initializations
    n_restarts = 8
    for restart in range(n_restarts):
        if restart == 0:
            pts0 = _fibonacci_sphere(N, seed=0)
        elif restart == 1:
            pts0 = _fibonacci_sphere(N, seed=1)
        elif restart == 2:
            pts0 = _fibonacci_sphere(N, seed=2)
        elif restart == 3:
            pts0 = _fibonacci_sphere(N, seed=3)
        else:
            # Random uniform points
            rng = np.random.RandomState(42 + restart)
            pts0 = rng.randn(N, 3)
            norms = np.linalg.norm(pts0, axis=1, keepdims=True)
            pts0 = pts0 / norms
        
        flat0 = _flatten(pts0)
        
        try:
            res = minimize(
                _objective,
                flat0,
                jac=_grad,
                method='L-BFGS-B',
                options={
                    'maxiter': 3000,
                    'ftol': 1e-15,
                    'gtol': 1e-12,
                    'maxls': 50
                }
            )
            pts_opt = _unflatten(res.x)
            norms = np.linalg.norm(pts_opt, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-15)
            pts_opt = pts_opt / norms
            
            E_opt = _energy(pts_opt)
            
            if E_opt < best_E:
                best_E = E_opt
                best_pts = pts_opt.copy()
        except Exception:
            continue
    
    # If L-BFGS-B failed entirely, fall back to Fibonacci
    if best_pts is None:
        best_pts = _fibonacci_sphere(N, seed=0)
    
    # Final polish with a few more iterations
    flat_best = _flatten(best_pts)
    try:
        res = minimize(
            _objective,
            flat_best,
            jac=_grad,
            method='L-BFGS-B',
            options={
                'maxiter': 5000,
                'ftol': 1e-16,
                'gtol': 1e-14,
                'maxls': 100
            }
        )
        pts_final = _unflatten(res.x)
        norms = np.linalg.norm(pts_final, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-15)
        pts_final = pts_final / norms
        E_final = _energy(pts_final)
        if E_final < best_E:
            best_pts = pts_final
    except Exception:
        pass
    
    return best_pts
# EVOLVE-BLOCK-END
