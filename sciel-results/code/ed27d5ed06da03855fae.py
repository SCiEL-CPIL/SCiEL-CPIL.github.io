"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
from scipy.optimize import minimize, LinearConstraint


def _energy_and_grad(pts):
    """Compute logarithmic energy sum_{i<j} log(1/|x_i - x_j|) and gradient.
    
    Energy = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = pts.shape[0]
    
    # Pairwise distances
    # diff[i,j] = pts[i] - pts[j], shape (n, n, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n, n, 3)
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)  # (n, n)
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero on diagonal
    
    dist = np.sqrt(dist_sq)
    np.fill_diagonal(dist, 1.0)
    
    # Energy: -sum_{i<j} log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: for each i, grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # = sum_{j != i} diff[i,j] / dist_sq[i,j]
    # diff[i,j] / dist_sq[i,j] -> shape (n, n, 3)
    grad_terms = diff / dist_sq[:, :, np.newaxis]  # (n, n, 3)
    # Set diagonal to 0
    grad_terms[np.arange(n), np.arange(n), :] = 0.0
    grad = np.sum(grad_terms, axis=1)  # (n, 3)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _energy_sphere(pts):
    """Energy on the sphere (points assumed on sphere or projected)."""
    pts = _project_to_sphere(pts)
    e, _ = _energy_and_grad(pts)
    return e


def construct():
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s
    
    # Golden-angle (Fibonacci) spiral points as initial guess
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Project to ensure unit sphere
    pts0 = _project_to_sphere(pts0)
    
    best_pts = pts0.copy()
    best_energy = _energy_sphere(best_pts)
    
    # Phase 1: Optimize in R^3 with soft constraint (penalty for deviation from unit sphere)
    # Use L-BFGS-B with a penalty term to keep points near the sphere
    
    def objective_and_grad(x_flat):
        pts = x_flat.reshape(N, 3)
        e, g = _energy_and_grad(pts)
        # Add penalty for deviation from unit sphere
        norms = np.linalg.norm(pts, axis=1)
        penalty = 1e4 * np.sum((norms - 1.0) ** 2)
        # Gradient of penalty: 2e4 * (norms - 1) * pts / norms^2
        penalty_grad = 2e4 * (norms - 1.0)[:, np.newaxis] * pts / np.maximum(norms[:, np.newaxis], 1e-15) ** 2
        total_e = e + penalty
        total_g = g + penalty_grad
        return total_e, total_g.flatten()
    
    x0 = pts0.flatten()
    
    # Run L-BFGS-B
    try:
        res = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        if res.success or res.fun < best_energy:
            pts_opt = _project_to_sphere(res.x.reshape(N, 3))
            e_opt = _energy_sphere(pts_opt)
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt
    except Exception:
        pass
    
    # Phase 2: Riemannian-style optimization on the sphere
    # Use gradient descent with retraction to sphere
    if time.time() < deadline - 5.0:
        pts = best_pts.copy()
        lr = 0.01
        for it in range(5000):
            if time.time() > deadline - 3.0:
                break
            e, g = _energy_and_grad(pts)
            # Project gradient to tangent space: g_tangent = g - (g·x)x
            g_proj = g - np.sum(pts * g, axis=1, keepdims=True) * pts
            pts_new = pts - lr * g_proj
            pts_new = _project_to_sphere(pts_new)
            e_new = _energy_sphere(pts_new)
            if e_new < e:
                pts = pts_new
                if e_new < best_energy:
                    best_energy = e_new
                    best_pts = pts.copy()
                # Adapt learning rate
                if it % 100 == 0:
                    lr = max(lr * 0.95, 1e-6)
            else:
                lr *= 0.5
                if lr < 1e-8:
                    break
        
        # Final polish with smaller steps
        lr = 1e-5
        for it in range(1000):
            if time.time() > deadline - 1.0:
                break
            e, g = _energy_and_grad(pts)
            g_proj = g - np.sum(pts * g, axis=1, keepdims=True) * pts
            pts_new = pts - lr * g_proj
            pts_new = _project_to_sphere(pts_new)
            e_new = _energy_sphere(pts_new)
            if e_new < e:
                pts = pts_new
                if e_new < best_energy:
                    best_energy = e_new
                    best_pts = pts.copy()
    
    # Phase 3: Try a few random restarts if time permits
    if time.time() < deadline - 10.0:
        rng = np.random.RandomState(42)
        for restart in range(5):
            if time.time() > deadline - 5.0:
                break
            # Perturb best solution
            noise = rng.randn(N, 3) * 0.1
            pts_r = _project_to_sphere(best_pts + noise)
            
            # Quick optimization
            lr = 0.005
            for it in range(2000):
                if time.time() > deadline - 2.0:
                    break
                e, g = _energy_and_grad(pts_r)
                g_proj = g - np.sum(pts_r * g, axis=1, keepdims=True) * pts_r
                pts_new = pts_r - lr * g_proj
                pts_new = _project_to_sphere(pts_new)
                e_new = _energy_sphere(pts_new)
                if e_new < e:
                    pts_r = pts_new
                    if e_new < best_energy:
                        best_energy = e_new
                        best_pts = pts_r.copy()
                else:
                    lr *= 0.5
                    if lr < 1e-8:
                        break
    
    # Final validation
    best_pts = _project_to_sphere(best_pts)
    
    # Ensure no points are too close
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < min_dist:
                min_dist = d
    
    if min_dist < 1e-9:
        # Fallback to Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        best_pts = _project_to_sphere(best_pts)
    
    # Ensure finite values
    if not np.all(np.isfinite(best_pts)):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
        best_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
