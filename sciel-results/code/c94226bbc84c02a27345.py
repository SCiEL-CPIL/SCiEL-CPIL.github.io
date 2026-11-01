import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
DEADLINE_SECONDS = 115.0  # Safety margin
EPS = 1e-12

def _energy_and_grad(pts):
    """
    pts: (N, 3) array of unit vectors.
    Returns (energy, grad) where energy = sum_{i<j} log(1/|xi-xj|)
    and grad is the gradient w.r.t. pts in R^{N x 3}.
    """
    # Pairwise differences: (N, N, 3)
    diff = pts[:, None, :] - pts[None, :, :]
    # Squared distances
    dist2 = np.sum(diff * diff, axis=2)
    # Mask diagonal
    mask = ~np.eye(N, dtype=bool)
    # Safe distances
    dist = np.sqrt(np.where(mask, np.maximum(dist2, EPS), 1.0))
    
    # Energy: sum_{i<j} log(1/dist) = -sum_{i<j} log(dist)
    # We sum over all i!=j and divide by 2
    log_terms = np.where(mask, -np.log(dist), 0.0)
    energy = 0.5 * np.sum(log_terms)
    
    # Gradient:
    # d/dp_i log(1/|p_i - p_j|) = -(p_i - p_j) / |p_i - p_j|^2
    # grad_i = sum_{j!=i} (p_j - p_i) / |p_i - p_j|^2
    # Note: dist2 is |p_i - p_j|^2
    inv_dist2 = np.where(mask, 1.0 / np.maximum(dist2, EPS), 0.0)
    
    # grad[i] = sum_j (pts[j] - pts[i]) * inv_dist2[i, j]
    # = sum_j pts[j] * inv_dist2[i, j] - pts[i] * sum_j inv_dist2[i, j]
    # Let's compute the sum of weights for each i
    weight_sums = np.sum(inv_dist2, axis=1, keepdims=True)
    # Sum of pts[j] * inv_dist2[i, j]
    # This is inv_dist2 @ pts
    weighted_pts = inv_dist2 @ pts
    
    grad = weighted_pts - weight_sums * pts
    
    return energy, grad

def _tangent_grad(pts, grad):
    """Project Euclidean gradient onto the tangent space."""
    radial = np.sum(grad * pts, axis=1, keepdims=True)
    return grad - radial * pts

def _retract(pts, step):
    """Move points along step and project back to sphere."""
    new_pts = pts + step
    norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
    # Avoid division by zero
    norms = np.where(norms < EPS, 1.0, norms)
    return new_pts / norms

def _optimize_sphere(pts0, max_iters=2000, tol=1e-10):
    """
    Riemannian Gradient Descent with Backtracking Line Search.
    """
    pts = pts0.copy()
    energy, grad = _energy_and_grad(pts)
    g_t = _tangent_grad(pts, grad)
    
    best_pts = pts.copy()
    best_energy = energy
    
    lr = 0.01
    for it in range(max_iters):
        # Check convergence
        if np.linalg.norm(g_t) < tol:
            break
            
        # Backtracking line search
        alpha = lr
        improved = False
        for _ in range(20):
            new_pts = _retract(pts, alpha * g_t)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy - 1e-4 * alpha * np.sum(g_t * g_t):
                pts = new_pts
                energy = new_energy
                improved = True
                break
            alpha *= 0.5
        
        if not improved:
            lr *= 0.5
            if lr < 1e-12:
                break
            continue
        
        # Update gradient
        _, grad = _energy_and_grad(pts)
        g_t = _tangent_grad(pts, grad)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            
        # Slowly decay learning rate
        lr = max(0.001, lr * 0.999)
        
    return best_pts, best_energy

def construct():
    t_start = time.time()
    
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
    
    # 2. Riemannian Gradient Descent
    pts_opt, energy_opt = _optimize_sphere(pts, max_iters=1000)
    
    # 3. L-BFGS-B Polish
    # We use the ambient gradient projection trick.
    # The objective function must be smooth.
    def fun_and_grad(x):
        x_pts = x.reshape(N, 3)
        x_pts = x_pts / np.linalg.norm(x_pts, axis=1, keepdims=True)
        e, g = _energy_and_grad(x_pts)
        g_t = _tangent_grad(x_pts, g)
        return e, g_t.ravel()
        
    x0 = pts_opt.ravel()
    
    # Check time
    if time.time() - t_start < DEADLINE_SECONDS - 10:
        try:
            res = minimize(fun_and_grad, x0, jac=True, method='L-BFGS-B',
                           options={'maxiter': 100, 'gtol': 1e-12, 'ftol': 1e-15})
            if res.success or res.fun < energy_opt:
                x_opt = res.x.reshape(N, 3)
                x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
                e_opt, _ = _energy_and_grad(x_opt)
                if e_opt < energy_opt:
                    pts_opt = x_opt
                    energy_opt = e_opt
        except Exception:
            pass
            
    # 4. Final Check and Return
    pts_opt = pts_opt / np.linalg.norm(pts_opt, axis=1, keepdims=True)
    return pts_opt
