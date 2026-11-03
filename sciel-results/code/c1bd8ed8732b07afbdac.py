"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 150
TIME_LIMIT = 115.0  # safety margin before 120s limit


def _log_energy_and_grad(pts):
    """
    Compute logarithmic energy and its gradient for points on the unit sphere.
    Energy = sum_{i<j} log(1 / |x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    
    pts: (N, 3) array of points on the unit sphere
    Returns: (energy, gradient) where gradient is the unconstrained gradient
             (caller must project to tangent space)
    """
    n = len(pts)
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (N, N, 3)
    dist = np.linalg.norm(diff, axis=2)  # (N, N)
    
    # Avoid self-interaction and division by zero
    np.fill_diagonal(dist, 1.0)
    
    # Energy: sum over i<j of -log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: d/dx_i [-log(|x_i - x_j|)] = (x_i - x_j) / |x_i - x_j|^2
    # For each point i, grad_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # Using the diff matrix: diff[i,j] = x_i - x_j
    # grad_i = sum_j (x_i - x_j) / dist_ij^2 = sum_j diff[i,j] / dist[i,j]^2
    
    # dist^2 matrix
    dist_sq = dist * dist
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero (diagonal doesn't contribute)
    
    # Weight matrix: W[i,j] = 1 / dist[i,j]^2 for i != j, 0 for i == j
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # grad_i = sum_j W[i,j] * diff[i,j] = sum_j (1/dist_ij^2) * (x_i - x_j)
    # = (sum_j W[i,j]) * x_i - sum_j W[i,j] * x_j
    # = row_sum_i * x_i - (W @ pts)_i
    
    row_sum = W.sum(axis=1)  # (N,)
    W_pts = W @ pts  # (N, 3)
    grad = row_sum[:, np.newaxis] * pts - W_pts  # (N, 3)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _tangent_project(pts, grad):
    """Project gradient onto tangent space at each point."""
    # tangent projection: g_t = g - (g . x) * x
    dot = np.sum(pts * grad, axis=1, keepdims=True)
    return grad - dot * pts


def _energy_callback(pts, best_state):
    """Callback for tracking best energy."""
    pts = _project_to_sphere(pts)
    energy, _ = _log_energy_and_grad(pts)
    if energy < best_state['best_energy']:
        best_state['best_energy'] = energy
        best_state['best_pts'] = pts.copy()


def _optimize_from_start(start_pts, time_limit):
    """Run L-BFGS-B optimization from a starting configuration."""
    best_state = {'best_energy': float('inf'), 'best_pts': start_pts.copy()}
    start_time = time.time()
    
    # Convert to flat parameterization: use spherical coordinates or just optimize in R^3 and project
    # We'll optimize in R^3 and project to sphere at each step
    x0 = start_pts.flatten()
    
    def objective(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        e, _ = _log_energy_and_grad(pts)
        return e
    
    def jac(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        _, g = _log_energy_and_grad(pts)
        # Project gradient to tangent space, then flatten
        g_t = _tangent_project(pts, g)
        return g_t.flatten()
    
    # Use L-BFGS-B
    result = minimize(
        objective, x0, jac=jac,
        method='L-BFGS-B',
        options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-10, 'maxls': 50}
    )
    
    if result.success or result.fun < best_state['best_energy']:
        best_pts = _project_to_sphere(result.x.reshape(N, 3))
        best_energy, _ = _log_energy_and_grad(best_pts)
        if best_energy < best_state['best_energy']:
            best_state['best_energy'] = best_energy
            best_state['best_pts'] = best_pts
    
    return best_state['best_pts'], best_state['best_energy']


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed):
    """Generate random points on the sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _perturbed_sphere(n, base_pts, sigma, seed):
    """Generate perturbed sphere points."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(n, 3) * sigma
    pts = base_pts + noise
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def construct():
    """Construct 150 points on the unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    
    # Best result so far
    best_pts = _fibonacci_sphere(N)
    best_energy, _ = _log_energy_and_grad(best_pts)
    
    # Generate diverse starting configurations
    starts = []
    
    # 1. Fibonacci sphere (baseline)
    starts.append(_fibonacci_sphere(N))
    
    # 2. Random starts
    for seed in [42, 123, 456, 789, 1011, 2022, 3033, 4044]:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        starts.append(_random_sphere(N, seed))
    
    # 3. Perturbed Fibonacci starts
    for sigma in [0.1, 0.3, 0.5]:
        for seed in [1, 2, 3]:
            if time.time() - start_time > TIME_LIMIT - 30:
                break
            starts.append(_perturbed_sphere(N, _fibonacci_sphere(N), sigma, seed))
    
    # Optimize from each start
    for i, start in enumerate(starts):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        try:
            pts, energy = _optimize_from_start(start, TIME_LIMIT - (time.time() - start_time))
            if energy < best_energy:
                best_energy = energy
                best_pts = pts
        except Exception:
            continue
    
    # Final projection to ensure unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    # Verify validity
    if not np.all(np.isfinite(best_pts)):
        return _fibonacci_sphere(N)
    
    # Check minimum distance
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, 1.0)
    min_dist = np.min(dist)
    if min_dist < 1e-9:
        return _fibonacci_sphere(N)
    
    return best_pts


# EVOLVE-BLOCK-START
# (The entire logic is above; this block marker is kept for compatibility)
# EVOLVE-BLOCK-END
