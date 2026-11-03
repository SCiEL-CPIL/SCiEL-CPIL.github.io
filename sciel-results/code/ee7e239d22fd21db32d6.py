"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _log_energy(points):
    """Compute log(1/d) energy for points on unit sphere."""
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=-1))
    np.fill_diagonal(dists, 1.0)
    iu = np.triu_indices(n, k=1)
    log_dists = np.log(dists[iu])
    return -np.sum(log_dists)


def _fibonacci_sphere(n):
    """Generate Fibonacci sphere points."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_uniform_sphere(n, seed=42):
    """Generate random uniform points on sphere."""
    rng = np.random.RandomState(seed)
    points = rng.randn(n, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms


def _project_to_sphere(points):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms


def _energy_and_grad(x):
    """Energy and gradient for L-BFGS-B. x is flat array of shape (N*3,)."""
    points = x.reshape(N, 3)
    points = _project_to_sphere(points)
    
    n = len(points)
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dists_sq, 1.0)
    
    # Energy: sum_{i<j} -log(d_ij)
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(np.sqrt(dists_sq[iu])))
    
    # Gradient: grad_i = sum_{j!=i} -(x_i - x_j) / d_ij^2
    weights = np.zeros_like(dists_sq)
    mask = ~np.eye(n, dtype=bool)
    weights[mask] = 1.0 / dists_sq[mask]
    
    sum_weights = np.sum(weights, axis=1)
    weighted_sum = weights @ points
    grads = points * sum_weights[:, np.newaxis] - weighted_sum
    
    # Project to tangent space
    dot_products = np.sum(grads * points, axis=1, keepdims=True)
    grads_tangent = grads - dot_products * points
    
    return energy, grads_tangent.reshape(-1)


def _optimize_lbfgsb(points, maxiter=2000, time_limit=60.0):
    """Optimize using L-BFGS-B with time limit."""
    start_time = time.time()
    x0 = points.reshape(-1).copy()
    
    def objective_and_grad(x):
        if time.time() - start_time > time_limit:
            raise RuntimeError("Time limit exceeded")
        e, g = _energy_and_grad(x)
        return e, g
    
    try:
        result = minimize(
            objective_and_grad,
            x0,
            jac=True,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': 1e-10, 'maxls': 20}
        )
        if result.success or result.fun < _log_energy(points):
            return _project_to_sphere(result.x.reshape(N, 3)), result.fun
    except Exception:
        pass
    
    return points, _log_energy(points)


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Start with multiple initializations
    initializations = [
        ("fibonacci", _fibonacci_sphere(N)),
        ("random1", _random_uniform_sphere(N, seed=42)),
        ("random2", _random_uniform_sphere(N, seed=123)),
        ("random3", _random_uniform_sphere(N, seed=456)),
    ]
    
    best_points = None
    best_energy = np.inf
    
    for name, init_points in initializations:
        if time.time() > deadline - 30:
            break
        
        # Use L-BFGS-B optimization
        opt_points, opt_energy = _optimize_lbfgsb(
            init_points, 
            maxiter=3000,
            time_limit=30.0
        )
        
        if opt_energy < best_energy:
            best_energy = opt_energy
            best_points = opt_points.copy()
    
    # Final polish with very tight tolerance if time permits
    if time.time() < deadline - 5 and best_points is not None:
        final_points, final_energy = _optimize_lbfgsb(
            best_points,
            maxiter=5000,
            time_limit=35.0
        )
        if final_energy < best_energy:
            best_energy = final_energy
            best_points = final_points.copy()
    
    # Ensure valid output
    if best_points is None:
        best_points = _fibonacci_sphere(N)
    
    best_points = _project_to_sphere(best_points)
    
    # Check for invalid points
    if not np.all(np.isfinite(best_points)):
        return _fibonacci_sphere(N)
    
    return best_points
# EVOLVE-BLOCK-END
