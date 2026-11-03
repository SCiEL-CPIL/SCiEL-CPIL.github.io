"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import math
import time

def _energy_fast(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    # pts is (N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    # Avoid self-distances (diagonal)
    np.fill_diagonal(dist, 1.0)
    # log(1/d) = -log(d)
    # Sum over upper triangle
    iu = np.triu_indices(N, k=1)
    vals = -np.log(dist[iu])
    return math.fsum(vals.tolist())

def _energy_grad(pts):
    """Compute energy and gradient for L-BFGS.
    Energy E = sum_{i<j} -log(||x_i - x_j||)
    dE/dx_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2  * (-1) * ... 
    Actually: E = sum_{i<j} -log(d_ij) where d_ij = ||x_i - x_j||
    dE/dx_i = sum_{j>i} (x_i - x_j)/d_ij^2 + sum_{j<i} (x_i - x_j)/d_ij^2
    = sum_{j != i} (x_i - x_j)/||x_i - x_j||^2
    """
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    np.fill_diagonal(dist, 1.0)
    
    # Gradient: for each i, sum over j != i of (x_i - x_j) / d_ij^2
    # = sum_j (x_i - x_j) / d_ij^2
    # (x_i - x_j)/d_ij^2 = diff[i,j] / dist_sq[i,j]
    grad = np.sum(diff / dist_sq[:, :, np.newaxis], axis=1)
    
    # Energy
    iu = np.triu_indices(N, k=1)
    e = math.fsum((-np.log(dist[iu])).tolist())
    
    return e, grad

def _project_to_sphere(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _validity_check(pts):
    """Check if points are valid: finite, nonzero, on sphere, min distance > 1e-9."""
    if not np.all(np.isfinite(pts)):
        return False
    if pts.shape != (N, 3):
        return False
    norms = np.linalg.norm(pts, axis=1)
    if np.any(norms < 1e-15):
        return False
    # Check min pairwise distance
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(dist[iu])
    if min_dist < 1e-9:
        return False
    return True

def _local_optimize(pts, maxiter=500, ftol=1e-16, gtol=1e-14):
    """Local optimization using L-BFGS-B with sphere constraint handled by projection."""
    def obj(x):
        pts = _project_to_sphere(x.reshape(N, 3))
        e, g = _energy_grad(pts)
        # Project gradient back to ambient space for unconstrained optimization
        # Simple approach: use the gradient in ambient space, projection handles constraint
        return e, g.flatten()
    
    try:
        res = minimize(obj, pts.flatten(), method='L-BFGS-B', 
                      jac=True, options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol,
                                         'maxls': 50, 'maxfun': 10000})
        best_pts = _project_to_sphere(res.x.reshape(N, 3))
        if _validity_check(best_pts):
            return best_pts
    except Exception:
        pass
    return pts

def _refine(pts, budget_seconds):
    """Multi-stage refinement with decreasing perturbation sizes."""
    start = time.time()
    current = pts.copy()
    current_energy = _energy_fast(current)
    
    # Stage 1: Coarse local optimization
    if time.time() - start > budget_seconds * 0.7:
        return current
    
    current = _local_optimize(current, maxiter=200, ftol=1e-14, gtol=1e-12)
    current_energy = _energy_fast(current)
    
    # Stage 2: Perturb and re-optimize
    rng = np.random.default_rng(42)
    for scale in [0.05, 0.01, 0.005, 0.001]:
        if time.time() - start > budget_seconds * 0.85:
            break
        for attempt in range(3):
            perturbed = current + rng.normal(0, scale, size=(N, 3))
            perturbed = _project_to_sphere(perturbed)
            if not _validity_check(perturbed):
                continue
            optimized = _local_optimize(perturbed, maxiter=300, ftol=1e-15, gtol=1e-13)
            e = _energy_fast(optimized)
            if e < current_energy:
                current = optimized
                current_energy = e
    
    # Stage 3: Ultra-precise polish
    if time.time() - start < budget_seconds * 0.9:
        current = _local_optimize(current, maxiter=1000, ftol=1e-18, gtol=1e-15)
        if _validity_check(current):
            current_energy = _energy_fast(current)
    
    return current

def _fibonacci_sphere():
    """Golden-angle Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    
    # Generate diverse initial configurations
    starts = []
    
    # 1. Fibonacci spiral
    starts.append(_fibonacci_sphere())
    
    # 2. Random uniform on sphere
    rng = np.random.default_rng(123)
    for _ in range(5):
        pts = rng.normal(size=(N, 3))
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        starts.append(pts)
    
    # 3. Clustered starts (some points close together)
    for _ in range(3):
        pts = rng.normal(size=(N, 3))
        # Create some clusters
        for c in range(3):
            center = pts[c] / np.linalg.norm(pts[c])
            for j in range(c + 1, min(c + 5, N)):
                pts[j] = center + rng.normal(0, 0.1, size=3)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        starts.append(pts)
    
    # 4. Grid-based starts
    # Approximate 10x10 grid on sphere using spherical coordinates
    for lat_div, lon_div in [(8, 12), (10, 10)]:
        lats = np.linspace(-np.pi/2 + np.pi/(2*lat_div), np.pi/2 - np.pi/(2*lat_div), lat_div)
        lons = np.linspace(0, 2*np.pi, lon_div, endpoint=False)
        pts_list = []
        for lat in lats:
            for lon in lons:
                x = np.cos(lat) * np.cos(lon)
                y = np.cos(lat) * np.sin(lon)
                z = np.sin(lat)
                pts_list.append([x, y, z])
        pts = np.array(pts_list[:N])
        if len(pts) == N:
            starts.append(pts)
    
    # Evaluate and refine each start
    best_pts = starts[0]
    best_energy = _energy_fast(best_pts)
    
    for idx, start_pts in enumerate(starts):
        if time.time() > deadline - 20.0:
            break
        if not _validity_check(start_pts):
            continue
        
        e0 = _energy_fast(start_pts)
        if e0 < best_energy:
            best_pts = start_pts
            best_energy = e0
        
        # Refine
        refined = _refine(start_pts, budget_seconds=25.0)
        e = _energy_fast(refined)
        if e < best_energy:
            best_pts = refined
            best_energy = e
    
    # Final polish on the best
    if time.time() < deadline - 5.0:
        final = _local_optimize(best_pts, maxiter=2000, ftol=1e-18, gtol=1e-16)
        if _validity_check(final):
            best_pts = final
    
    # Ensure valid output
    best_pts = _project_to_sphere(best_pts)
    if not _validity_check(best_pts):
        # Fallback to Fibonacci
        best_pts = _fibonacci_sphere()
    
    return best_pts
# EVOLVE-BLOCK-END
