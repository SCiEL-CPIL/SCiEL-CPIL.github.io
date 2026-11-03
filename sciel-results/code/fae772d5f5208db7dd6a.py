"""Optimize logarithmic energy of 150 points on the unit sphere."""
import numpy as np
import math
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

N = 150
DEADLINE = 115.0  # seconds, with 5s safety margin before 120s limit


def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    where d_ij is chordal distance.
    
    Gradient: dE/dx_i = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    # Compute pairwise distances
    dist = cdist(pts, pts, 'euclidean')
    np.fill_diagonal(dist, 1.0)  # avoid log(0)
    
    # Energy: sum over i<j of -log(d_ij)
    iu = np.triu_indices(n, k=1)
    dist_upper = dist[iu]
    # Mask very small distances to avoid -inf or huge values
    dist_upper = np.maximum(dist_upper, 1e-15)
    energy = -np.sum(np.log(dist_upper))
    
    # Gradient: dE/dx_i = -sum_{j!=i} (x_i - x_j) / dist_sq[i,j]
    # = sum_{j!=i} (x_j - x_i) / dist_sq[i,j]
    
    dist_sq = dist ** 2
    inv_dist_sq = np.zeros_like(dist_sq)
    mask = dist_sq > 1e-30
    inv_dist_sq[mask] = 1.0 / dist_sq[mask]
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    row_sums = np.sum(inv_dist_sq, axis=1)  # (N,)
    weighted_sum = inv_dist_sq @ pts  # (N, 3)
    
    # grad[i] = sum_j (x_j - x_i) / dist_sq[i,j] = weighted_sum[i] - x_i * row_sums[i]
    grad = weighted_sum - pts * row_sums[:, np.newaxis]  # (N, 3)
    
    return energy, grad


def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_sphere(n, seed=None):
    """Generate Fibonacci spiral points on unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
        offset = rng.uniform(0, 1)
    else:
        offset = 0.0
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _icosahedron_based(n):
    """Generate points using icosahedron subdivision for better initial distribution."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    ico = ico / np.linalg.norm(ico, axis=1, keepdims=True)
    
    # If n == 12, return icosahedron
    if n == 12:
        return ico
    
    # Use Fibonacci for the rest but perturb near icosahedron
    fib = _fibonacci_sphere(n)
    result = fib.copy()
    for i in range(12):
        # Find closest point in fib to each ico vertex
        dists = np.linalg.norm(fib - ico[i], axis=1)
        closest = np.argmin(dists)
        # Pull it toward the icosahedron vertex
        result[closest] = ico[i]
    return _project_to_sphere(result)


def _random_sphere(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    return _project_to_sphere(pts)


def _so2_rotate(pts, rng):
    """Apply random SO(3) rotation to points."""
    # Generate random rotation matrix
    u = rng.standard_normal(3)
    v = rng.standard_normal(3)
    w = rng.standard_normal(3)
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    w = np.cross(u, v)
    w = w / np.linalg.norm(w)
    R = np.column_stack([u, v, w])
    return pts @ R.T


def _lbfgs_optimize(pts, time_limit=10.0, ftol=1e-12, gtol=1e-10, maxiter=2000):
    """L-BFGS-B optimization with time limit."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())

    def objective_flat(x):
        if time.time() - start_time > time_limit:
            return np.inf
        p = _project_to_sphere(x.reshape(-1, 3))
        e, _ = _energy_and_grad(p)
        return e

    def grad_flat(x):
        if time.time() - start_time > time_limit:
            return np.zeros(len(x))
        p = _project_to_sphere(x.reshape(-1, 3))
        _, g = _energy_and_grad(p)
        # Project gradient to tangent space
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return g_tangent.flatten()

    x0 = pts.flatten()

    try:
        result = minimize(
            objective_flat,
            x0,
            jac=grad_flat,
            method='L-BFGS-B',
            options={
                'maxiter': maxiter,
                'maxfun': 5000,
                'ftol': ftol,
                'gtol': gtol,
                'maxls': 50
            }
        )
        final_pts = _project_to_sphere(result.x.reshape(-1, 3))
        energy, _ = _energy_and_grad(final_pts)
        return final_pts, energy
    except Exception:
        return pts, float('inf')


def _polish(pts, time_limit=10.0):
    """Final high-precision polish with very tight tolerances."""
    start_time = time.time()
    pts = _project_to_sphere(pts.copy())

    def objective_flat(x):
        if time.time() - start_time > time_limit:
            return np.inf
        p = _project_to_sphere(x.reshape(-1, 3))
        e, _ = _energy_and_grad(p)
        return e

    def grad_flat(x):
        if time.time() - start_time > time_limit:
            return np.zeros(len(x))
        p = _project_to_sphere(x.reshape(-1, 3))
        _, g = _energy_and_grad(p)
        g_tangent = g - np.sum(g * p, axis=1, keepdims=True) * p
        return g_tangent.flatten()

    x0 = pts.flatten()

    try:
        result = minimize(
            objective_flat,
            x0,
            jac=grad_flat,
            method='L-BFGS-B',
            options={
                'maxiter': 5000,
                'maxfun': 10000,
                'ftol': 1e-15,
                'gtol': 1e-12,
                'maxls': 100
            }
        )
        final_pts = _project_to_sphere(result.x.reshape(-1, 3))
        energy, _ = _energy_and_grad(final_pts)
        return final_pts, energy
    except Exception:
        return pts, float('inf')


def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    # Generate diverse starting points
    starts = []
    
    # 1. Fibonacci spiral
    starts.append(_fibonacci_sphere(N))
    
    # 2. Icosahedron-based
    starts.append(_icosahedron_based(N))
    
    # 3. Perturbed Fibonacci
    for _ in range(3):
        fib = _fibonacci_sphere(N)
        noise = 0.05 * rng.standard_normal((N, 3))
        perturbed = fib + noise
        starts.append(_project_to_sphere(perturbed))
    
    # 4. Random points
    for _ in range(4):
        starts.append(_random_sphere(N, rng))
    
    # 5. Rotated Fibonacci
    for _ in range(2):
        fib = _fibonacci_sphere(N)
        starts.append(_so2_rotate(fib, rng))
    
    # 6. Clustered starts
    for _ in range(2):
        centers = _random_sphere(10, rng)
        pts = []
        for c in centers:
            local = c + 0.1 * rng.standard_normal((15, 3))
            pts.append(_project_to_sphere(local))
        starts.append(np.vstack(pts))
    
    # Optimize each start
    best_pts = None
    best_energy = float('inf')
    
    n_starts = len(starts)
    # Allocate time: ~80% for initial optimizations, ~20% for final polish
    opt_budget = (DEADLINE - (time.time() - start_time)) * 0.8
    polish_budget = (DEADLINE - (time.time() - start_time)) * 0.2
    
    time_per_opt = opt_budget / n_starts
    time_per_opt = max(2.0, min(10.0, time_per_opt))
    
    for i, s in enumerate(starts):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        remaining = DEADLINE - (time.time() - start_time)
        if remaining < 15:
            break
        
        opt_time = min(time_per_opt, remaining - 10)
        if opt_time < 2:
            break
        
        try:
            pts_opt, e_opt = _lbfgs_optimize(s, opt_time)
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_pts = pts_opt.copy()
        except Exception:
            continue
    
    # Final high-precision polish
    if best_pts is not None and time.time() - start_time < DEADLINE - 5:
        remaining = DEADLINE - (time.time() - start_time) - 5
        if remaining > 3:
            try:
                pts_final, e_final = _polish(best_pts, remaining)
                if e_final < best_energy:
                    best_energy = e_final
                    best_pts = pts_final.copy()
            except Exception:
                pass
    
    if best_pts is None:
        # Fallback to Fibonacci
        best_pts = _fibonacci_sphere(N)
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    assert best_pts.shape == (N, 3)
    assert np.all(np.isfinite(best_pts))
    assert np.all(np.linalg.norm(best_pts, axis=1) > 1e-15)
    
    # Check no two points are too close
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                # Separate them slightly
                best_pts[j] += 1e-8 * rng.standard_normal(3)
                best_pts[j] = best_pts[j] / np.linalg.norm(best_pts[j])
    
    return best_pts
