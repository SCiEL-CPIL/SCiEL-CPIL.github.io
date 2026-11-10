import numpy as np
import time
import math

N = 150

def _log_energy_and_grad(points):
    """Compute log energy and its gradient for points on unit sphere (vectorized)."""
    n = points.shape[0]
    # Compute pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero for i=j
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Energy: sum_{i<j} log(1/dist_ij) = -sum_{i<j} log(dist_ij)
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu[0], iu[1]]))
    
    # Vectorized gradient computation
    # For pair (i,j): grad[i] += -(x_i - x_j)/|x_i - x_j|^2, grad[j] += +(x_i - x_j)/|x_i - x_j|^2
    # coeff[i,j] = -1/dist_sq[i,j] for i != j
    coeff = np.zeros_like(dist_sq)
    mask = ~np.eye(n, dtype=bool)
    coeff[mask] = -1.0 / dist_sq[mask]
    
    # grad[i] = sum_j coeff[i,j] * (x_i - x_j) = sum_j coeff[i,j]*x_i - sum_j coeff[i,j]*x_j
    # = x_i * sum_j coeff[i,j] - (coeff @ points)[i]
    row_sums = coeff.sum(axis=1)  # shape (n,)
    grad = points * row_sums[:, np.newaxis] - coeff @ points
    
    return energy, grad


def _project_to_sphere(points):
    """Project points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return points / norms


def _log_energy_fast(points):
    """Fast energy computation for tracking."""
    n = points.shape[0]
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    iu = np.triu_indices(n, k=1)
    return -np.sum(np.log(dist[iu[0], iu[1]]))


def _fibonacci_spiral(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, seed=None):
    """Generate random points on unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin before 120s limit
    
    from scipy.optimize import minimize
    
    def objective_and_grad(x_flat):
        pts = x_flat.reshape(-1, 3)
        pts = _project_to_sphere(pts)
        e, g = _log_energy_and_grad(pts)
        # Project gradient to tangent space: g_tangent = g - (g·p)p
        dot = np.sum(g * pts, axis=1, keepdims=True)
        g_tangent = g - dot * pts
        return e, g_tangent.reshape(-1)
    
    # Generate multiple initial configurations
    inits = []
    inits.append(_fibonacci_spiral(N))
    # Add several random starts with different seeds
    for s in range(5):
        inits.append(_random_points(N, seed=42 + s))
    
    best_points = None
    best_energy = np.inf
    
    # Phase 1: Optimize each initial configuration
    for init_idx, init_pts in enumerate(inits):
        if time.time() > deadline - 20.0:
            break
        
        try:
            result = minimize(
                objective_and_grad,
                init_pts.reshape(-1),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50}
            )
            
            if result.x is not None and np.all(np.isfinite(result.x)):
                cand = _project_to_sphere(result.x.reshape(-1, 3))
                cand_energy = _log_energy_fast(cand)
                if cand_energy < best_energy:
                    best_energy = cand_energy
                    best_points = cand.copy()
        except Exception:
            pass
    
    # Phase 2: Polish the best solution with tighter tolerances
    if best_points is not None and time.time() < deadline - 10.0:
        try:
            result2 = minimize(
                objective_and_grad,
                best_points.reshape(-1),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 50}
            )
            if result2.x is not None and np.all(np.isfinite(result2.x)):
                cand2 = _project_to_sphere(result2.x.reshape(-1, 3))
                cand2_energy = _log_energy_fast(cand2)
                if cand2_energy < best_energy:
                    best_energy = cand2_energy
                    best_points = cand2.copy()
        except Exception:
            pass
    
    # Phase 3: Final ultra-tight polish if time permits
    if best_points is not None and time.time() < deadline - 3.0:
        try:
            result3 = minimize(
                objective_and_grad,
                best_points.reshape(-1),
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 30}
            )
            if result3.x is not None and np.all(np.isfinite(result3.x)):
                cand3 = _project_to_sphere(result3.x.reshape(-1, 3))
                cand3_energy = _log_energy_fast(cand3)
                if cand3_energy < best_energy:
                    best_energy = cand3_energy
                    best_points = cand3.copy()
        except Exception:
            pass
    
    # Final projection and validation
    if best_points is None:
        best_points = _fibonacci_spiral(N)
    
    best_points = _project_to_sphere(best_points)
    
    # Ensure no points are too close
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                best_points[j] += 1e-8 * np.random.randn(3)
                best_points = _project_to_sphere(best_points)
    
    return best_points
