"""
Optimized 50-point spherical covering using differentiable Voronoi-based optimization.

Approach:
- Represent configuration as 50 points on the unit sphere.
- For each point, compute its Voronoi cell inradius (max distance to cell vertices).
- Cell vertices are circumcenters of Delaunay triangles (triples of neighbors).
- Minimize sum of squared inradii using gradient-based optimization.
- Use smooth approximations to handle non-differentiability at cell boundaries.
- Vectorized circumradius computation for speed.
- Multi-scale soft-max temperature for robust optimization.
"""

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _normalize(v):
    """Normalize vectors to unit length."""
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return v / norms


def _compute_circumradii_vectorized(points, triples):
    """
    Vectorized computation of spherical circumradii for many triples.
    
    Parameters:
    - points: (N, 3) array of unit vectors
    - triples: (M, 3) array of integer indices, each row is (i, j, k)
    
    Returns:
    - (M,) array of circumradii (angular distances)
    """
    if len(triples) == 0:
        return np.array([])
    
    p1 = points[triples[:, 0]]  # (M, 3)
    p2 = points[triples[:, 1]]  # (M, 3)
    p3 = points[triples[:, 2]]  # (M, 3)
    
    # Compute circumcenter direction: cross(p1-p2, p1-p3)
    v1 = p1 - p2  # (M, 3)
    v2 = p1 - p3  # (M, 3)
    c = np.cross(v1, v2)  # (M, 3)
    
    norms = np.linalg.norm(c, axis=1, keepdims=True)  # (M, 1)
    # Handle degenerate cases
    valid = norms.squeeze() > 1e-10
    c_safe = c.copy()
    c_safe[valid] = c[valid] / norms[valid]
    
    # For degenerate cases, use midpoint as approximation
    degenerate = ~valid
    if np.any(degenerate):
        midpoints = (p1[degenerate] + p2[degenerate] + p3[degenerate]) / 3.0
        mid_norms = np.linalg.norm(midpoints, axis=1, keepdims=True)
        mid_norms = np.where(mid_norms < 1e-12, 1.0, mid_norms)
        c_safe[degenerate] = midpoints / mid_norms
    
    # Compute cos(R) = c · p1
    cos_r = np.sum(c_safe * p1, axis=1)
    cos_r = np.clip(cos_r, -1.0, 1.0)
    radii = np.arccos(cos_r)
    
    # For degenerate cases, use max pairwise distance / 2
    if np.any(degenerate):
        d12 = np.arccos(np.clip(np.sum(p1[degenerate] * p2[degenerate], axis=1), -1, 1))
        d13 = np.arccos(np.clip(np.sum(p1[degenerate] * p3[degenerate], axis=1), -1, 1))
        d23 = np.arccos(np.clip(np.sum(p2[degenerate] * p3[degenerate], axis=1), -1, 1))
        radii[degenerate] = np.maximum(np.maximum(d12, d13), d23) / 2.0
    
    return radii


def _get_delaunay_triples(points, k_neighbors=10):
    """
    Get relevant Delaunay triples for each point's Voronoi cell.
    
    For each point i, consider its k nearest neighbors.
    Generate all triples (i, j, k) where j, k are neighbors of i.
    
    Returns:
    - triples: (M, 3) array of integer indices
    - cell_indices: (M,) array mapping each triple to its center point
    """
    n = len(points)
    cos_dists = np.clip(points @ points.T, -1.0, 1.0)
    dists = np.arccos(cos_dists)
    np.fill_diagonal(dists, 0.0)
    
    all_triples = []
    all_cell_indices = []
    
    for i in range(n):
        # Get k nearest neighbors (excluding self)
        idx = np.argsort(dists[i])[:k_neighbors]
        
        # Generate all pairs of neighbors
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                j = idx[a]
                k = idx[b]
                all_triples.append((i, j, k))
                all_cell_indices.append(i)
    
    if len(all_triples) == 0:
        return np.empty((0, 3), dtype=int), np.empty(0, dtype=int)
    
    triples = np.array(all_triples, dtype=int)
    cell_indices = np.array(all_cell_indices, dtype=int)
    
    return triples, cell_indices


def _voronoi_inradii(points, k_neighbors=10, temperature=0.03):
    """
    Compute the inradius (covering radius contribution) for each point.
    
    For each point i, consider its k nearest neighbors. For each triple of neighbors
    (including i), compute the circumradius. The inradius of point i's Voronoi cell
    is the maximum of these circumradii over all relevant triples.
    
    We use a soft-max approximation for differentiability.
    """
    n = len(points)
    triples, cell_indices = _get_delaunay_triples(points, k_neighbors)
    
    if len(triples) == 0:
        return np.zeros(n)
    
    # Compute all circumradii at once (vectorized)
    radii = _compute_circumradii_vectorized(points, triples)
    
    # For each cell, compute soft-max of its radii
    inradii = np.zeros(n)
    for i in range(n):
        mask = cell_indices == i
        if np.any(mask):
            cell_radii = radii[mask]
            # Soft-max for differentiable max
            inradii[i] = temperature * np.log(np.sum(np.exp(cell_radii / temperature)))
        else:
            inradii[i] = 0.0
    
    return inradii


def _objective(x, temperature=0.03):
    """
    Objective: sum of squared Voronoi inradii.
    This is differentiable and directly targets the covering radius.
    """
    points = _normalize(x.reshape(N, 3))
    inradii = _voronoi_inradii(points, temperature=temperature)
    return np.sum(inradii ** 2)


def _covering_radius(points):
    """Compute the actual covering radius for evaluation."""
    points = _normalize(points)
    # Use a small temperature for accurate max
    inradii = _voronoi_inradii(points, k_neighbors=10, temperature=0.001)
    return np.max(inradii) * 180.0 / np.pi


def _initial_fibonacci():
    """Generate initial Fibonacci spiral configuration."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _initial_known_solution():
    """
    Generate initial configuration based on known good structure.
    Uses a modified Fibonacci with slight perturbation to break symmetry.
    """
    points = _initial_fibonacci()
    # Add small deterministic perturbation to help optimization escape local minima
    rng = np.random.RandomState(12345)
    points = points + rng.randn(N, 3) * 0.02
    return _normalize(points)


def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    Uses differentiable Voronoi inradius optimization with multi-scale temperature.
    """
    start_time = time.time()
    max_time = 110  # Leave margin for safety
    
    # Initial configuration
    points = _initial_known_solution()
    x0 = points.flatten().copy()
    
    best_x = x0.copy()
    best_obj = _objective(x0, temperature=0.03)
    
    # Phase 1: Coarse optimization with larger temperature
    def obj_coarse(x):
        return _objective(x, temperature=0.05)
    
    try:
        result = minimize(
            obj_coarse,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 6000, 'ftol': 1e-8}
        )
        if result.fun < best_obj:
            best_x = result.x
            best_obj = _objective(result.x, temperature=0.03)
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > max_time - 40:
        return _normalize(best_x.reshape(N, 3))
    
    # Phase 2: Medium temperature optimization
    def obj_medium(x):
        return _objective(x, temperature=0.02)
    
    try:
        result = minimize(
            obj_medium,
            best_x,
            method='L-BFGS-B',
            options={'maxiter': 300, 'maxfun': 10000, 'ftol': 1e-10}
        )
        if result.fun < best_obj:
            best_x = result.x
            best_obj = _objective(result.x, temperature=0.03)
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > max_time - 30:
        return _normalize(best_x.reshape(N, 3))
    
    # Phase 3: Local search with multiple restarts
    rng = np.random.RandomState(42)
    for restart in range(6):
        # Create perturbation with decreasing scale
        scale = 0.10 * (1.0 - restart / 8.0)
        x_perturbed = best_x + rng.randn(N * 3) * scale
        try:
            result = minimize(
                obj_medium,
                x_perturbed,
                method='L-BFGS-B',
                options={'maxiter': 150, 'maxfun': 4000, 'ftol': 1e-10}
            )
            if result.fun < best_obj:
                best_x = result.x
                best_obj = _objective(result.x, temperature=0.03)
        except Exception:
            pass
        
        if time.time() - start_time > max_time - 20:
            break
    
    # Phase 4: Final fine-tuning with small temperature
    if time.time() - start_time < max_time - 10:
        def obj_fine(x):
            return _objective(x, temperature=0.005)
        
        try:
            result = minimize(
                obj_fine,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 400, 'maxfun': 12000, 'ftol': 1e-12}
            )
            if result.fun < best_obj:
                best_x = result.x
                best_obj = _objective(result.x, temperature=0.03)
        except Exception:
            pass
    
    points = _normalize(best_x.reshape(N, 3))
    
    # Validate: ensure points are on unit sphere and not degenerate
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms < 1e-6) or np.any(~np.isfinite(points)):
        # Fallback to Fibonacci
        points = _initial_fibonacci()
    
    # Final check: ensure no coincident points
    cos_dists = np.clip(points @ points.T, -1.0, 1.0)
    np.fill_diagonal(cos_dists, 0.0)
    min_dist = np.arccos(np.clip(1.0 - np.min(1.0 - cos_dists), -1, 1))
    if min_dist < 0.01:  # Less than ~0.57 degrees
        points = _initial_fibonacci()
    
    return points

# EVOLVE-BLOCK-END
