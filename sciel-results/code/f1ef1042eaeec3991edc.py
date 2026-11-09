"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 100

# EVOLVE-BLOCK-START
from scipy.optimize import minimize


def _log_energy(pts):
    """Compute log(1/d) energy sum over all pairs using chordal distances."""
    # pts: (N, 3)
    # Compute pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Get upper triangle indices
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    # Clip to avoid log(0)
    d = np.clip(d, 1e-15, None)
    # Sum log(1/d) = -sum(log(d))
    terms = [-math.log(di) for di in d]
    return math.fsum(terms)


def _log_energy_and_grad(pts):
    """Compute log(1/d) energy and gradient."""
    # pts: (N, 3)
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Avoid zero distances
    dist = np.clip(dist, 1e-15, None)
    
    # Energy: sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    iu = np.triu_indices(N, k=1)
    d = dist[iu]
    energy = -math.fsum([math.log(di) for di in d])
    
    # Gradient: d/dp_i [log(1/d_ij)] = -1/d_ij * d(d_ij)/dp_i
    # d(d_ij)/dp_i = (p_i - p_j) / d_ij
    # So d/dp_i [log(1/d_ij)] = -(p_i - p_j) / d_ij^2
    # For the sum over j != i: grad_i = sum_{j != i} (p_j - p_i) / d_ij^2
    
    # Vectorized gradient computation
    # diff[i,j] = p_i - p_j
    # We want sum_j (p_j - p_i) / d_ij^2 for each i
    # = -sum_j diff[i,j] / d_ij^2
    
    # Create weight matrix: W[i,j] = 1/d_ij^2 for i != j, 0 otherwise
    inv_dist_sq = np.zeros_like(dist)
    mask = ~np.eye(N, dtype=bool)
    inv_dist_sq[mask] = 1.0 / (dist[mask] ** 2)
    
    # grad_i = -sum_j W[i,j] * diff[i,j] = -sum_j W[i,j] * (p_i - p_j)
    # = -W[i,:] @ p_i * sum_j W[i,j] + sum_j W[i,j] * p_j
    # Actually: grad_i = sum_j W[i,j] * (p_j - p_i) = (W[i,:] @ pts) - p_i * sum_j W[i,j]
    
    # W @ pts gives for each i: sum_j W[i,j] * p_j
    weighted_sum = inv_dist_sq @ pts
    # sum_j W[i,j] for each i
    row_sums = inv_dist_sq.sum(axis=1)
    
    # grad_i = weighted_sum[i] - p_i * row_sums[i]
    grad = weighted_sum - pts * row_sums[:, None]
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-15, None)
    return pts / norms


def _initial_fibonacci():
    """Golden-angle (Fibonacci) spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _optimize_from_start(start, deadline):
    """Optimize from a starting configuration using L-BFGS-B with sphere constraint."""
    # Flatten to 1D for scipy
    x0 = start.flatten()
    
    def objective(x):
        pts = x.reshape(N, 3)
        # Project to sphere
        pts = _project_to_sphere(pts)
        e, g = _log_energy_and_grad(pts)
        return e, g
    
    # Use L-BFGS-B
    best_pts = start.copy()
    best_energy = _log_energy(start)
    
    try:
        result = minimize(
            objective,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxfun': 10000}
        )
        if result.success or result.fun < best_energy:
            result_pts = _project_to_sphere(result.x.reshape(N, 3))
            result_energy = _log_energy(result_pts)
            if result_energy < best_energy:
                best_pts = result_pts
                best_energy = result_energy
    except Exception:
        pass
    
    return best_pts, best_energy


def construct():
    """Construct 100 points on the unit sphere to minimize logarithmic energy."""
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s safety margin for 120s limit
    
    # Start with Fibonacci spiral
    pts = _initial_fibonacci()
    
    # Multiple restarts with different initializations
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    # Optimization from Fibonacci start
    if time.time() < deadline - 5:
        pts_opt, energy_opt = _optimize_from_start(pts, deadline)
        if energy_opt < best_energy:
            best_pts = pts_opt
            best_energy = energy_opt
    
    # Try a few random restarts if time permits
    rng = np.random.default_rng(42)
    for restart in range(5):
        if time.time() >= deadline - 10:
            break
        
        # Random initial points on sphere
        random_pts = rng.standard_normal((N, 3))
        random_pts = random_pts / np.linalg.norm(random_pts, axis=1, keepdims=True)
        
        # Optimize
        pts_opt, energy_opt = _optimize_from_start(random_pts, deadline)
        if energy_opt < best_energy:
            best_pts = pts_opt
            best_energy = energy_opt
    
    # Final projection to ensure unit sphere
    best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
