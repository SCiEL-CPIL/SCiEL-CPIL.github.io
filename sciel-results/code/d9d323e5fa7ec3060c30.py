"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize, L_BFGS_B

N = 100
DEADLINE = 145.0  # safety margin before 150s kill


def _energy_fsum(pts):
    """Compute log(1/d) energy using math.fsum for accuracy."""
    import math
    n = len(pts)
    terms = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-15:
                return -1e18
            terms.append(math.log(1.0 / d))
    return math.fsum(terms)


def _energy_and_grad(pts):
    """Compute energy and gradient. pts shape (N,3) on unit sphere."""
    n = pts.shape[0]
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N,N,3)
    dist = np.sqrt(np.sum(diff ** 2, axis=2))  # (N,N)
    np.fill_diagonal(dist, 1.0)  # avoid log(0)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    energy = -0.5 * np.sum(np.log(dist))
    
    # Gradient: d/dx_i sum_{j!=i} log(1/|x_i-x_j|) = sum_{j!=i} (x_i - x_j)/|x_i-x_j|^2
    # Note: log(1/d) = -log(d), so d(-log(d))/dx = -1/d * dd/dx
    # dd/dx_i = (x_i - x_j)/d
    # So d(-log(d))/dx_i = -(x_i - x_j)/d^2
    # But we want sum over j of d(log(1/d_ij))/dx_i
    # log(1/d) = -log(d)
    # d(-log(d))/dx_i = -(1/d) * (x_i - x_j)/d = -(x_i-x_j)/d^2
    # Wait, let me redo: energy = sum_{i<j} log(1/|xi-xj|) = -sum_{i<j} log|xi-xj|
    # d/dxi of -log|xi-xj| = -(xi-xj)/|xi-xj|^2
    # So gradient for point i is sum_{j!=i} -(xi-xj)/|xi-xj|^2 = sum_{j!=i} (xj-xi)/|xi-xj|^2
    
    # Using the diff matrix: diff[i,j] = xi - xj
    # grad_i = sum_j (xj - xi) / |xi-xj|^2 = -sum_j diff[i,j] / dist[i,j]^2
    inv_dist_sq = 1.0 / (dist ** 2)
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = -sum_j diff[i,j] * inv_dist_sq[i,j]
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def construct():
    t_start = time.time()
    
    # Initial: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = x0.copy()
    best_energy = _energy_fsum(best_pts)
    
    def sphere_fun(flat):
        """Objective: energy + penalty for off-sphere."""
        pts = flat.reshape(N, 3)
        # Penalty for not being on unit sphere
        norms = np.linalg.norm(pts, axis=1)
        penalty = 1e6 * np.sum((norms - 1.0) ** 2)
        e, _ = _energy_and_grad(pts)
        return e + penalty
    
    def sphere_fun_grad(flat):
        pts = flat.reshape(N, 3)
        e, g = _energy_and_grad(pts)
        # Penalty gradient
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        penalty_grad = 4e6 * (norms - 1.0) * pts / np.maximum(norms, 1e-15)
        return (g + penalty_grad).flatten()
    
    # Multiple restarts with perturbation
    rng = np.random.RandomState(42)
    
    for restart in range(20):
        if time.time() > DEADLINE:
            break
        
        if restart == 0:
            x0_curr = x0.copy()
        else:
            # Perturb from best
            noise_scale = 0.1 / (1.0 + restart * 0.5)
            x0_curr = best_pts + rng.randn(N, 3) * noise_scale
            x0_curr = _project_to_sphere(x0_curr)
        
        flat0 = x0_curr.flatten()
        
        try:
            res = minimize(
                sphere_fun, flat0,
                jac=sphere_fun_grad,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10, 'maxls': 50}
            )
        except Exception:
            continue
        
        if time.time() > DEADLINE:
            break
        
        pts = _project_to_sphere(res.x.reshape(N, 3))
        
        # Check validity
        if not np.all(np.isfinite(pts)):
            continue
        min_norm = np.min(np.linalg.norm(pts, axis=1))
        if min_norm < 1e-15:
            continue
        
        # Check minimum distance
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=2))
        np.fill_diagonal(dists, 1.0)
        min_dist = np.min(dists)
        if min_dist < 1e-9:
            continue
        
        e = _energy_fsum(pts)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
    
    # Final polish: a few more iterations from best
    if time.time() < DEADLINE - 5:
        flat0 = best_pts.flatten()
        try:
            res = minimize(
                sphere_fun, flat0,
                jac=sphere_fun_grad,
                method='L-BFGS-B',
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 100}
            )
            pts = _project_to_sphere(res.x.reshape(N, 3))
            if np.all(np.isfinite(pts)):
                e = _energy_fsum(pts)
                if e < best_energy:
                    best_energy = e
                    best_pts = pts.copy()
        except Exception:
            pass
    
    return _project_to_sphere(best_pts)
