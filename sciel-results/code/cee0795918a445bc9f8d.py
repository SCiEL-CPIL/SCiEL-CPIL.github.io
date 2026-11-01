"""
Optimized 50-point spherical covering using differentiable Voronoi-based optimization.

Approach:
- Represent configuration as 50 points on the unit sphere.
- For each point, compute its Voronoi cell inradius (max distance to cell vertices).
- Cell vertices are circumcenters of Delaunay triangles (triples of neighbors).
- Minimize sum of squared inradii using gradient-based optimization.
- Use smooth approximations to handle non-differentiability at cell boundaries.
"""

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time


def _normalize(v):
    """Normalize vectors to unit length."""
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return v / norms


def _spherical_circumcenter(p1, p2, p3):
    """
    Compute the spherical circumcenter of triangle (p1, p2, p3).
    The circumcenter c satisfies c·p1 = c·p2 = c·p3 = cos(R).
    Returns (c, R) where c is the circumcenter and R is the circumradius.
    """
    # The circumcenter lies in the plane perpendicular to the triangle's plane
    # passing through the origin. We solve for c such that c·p1 = c·p2 = c·p3.
    # This means c is perpendicular to (p1-p2) and (p1-p3).
    v1 = p1 - p2
    v2 = p1 - p3
    c = np.cross(v1, v2)
    norm_c = np.linalg.norm(c)
    if norm_c < 1e-12:
        # Degenerate case
        d12 = np.arccos(np.clip(np.dot(p1, p2), -1, 1))
        d13 = np.arccos(np.clip(np.dot(p1, p3), -1, 1))
        d23 = np.arccos(np.clip(np.dot(p2, p3), -1, 1))
        # Return midpoint of longest edge as approximate center
        if d12 >= d13 and d12 >= d23:
            mid = (p1 + p2) / np.linalg.norm(p1 + p2) if np.linalg.norm(p1 + p2) > 1e-12 else p1
            return mid, d12 / 2.0
        elif d13 >= d12 and d13 >= d23:
            mid = (p1 + p3) / np.linalg.norm(p1 + p3) if np.linalg.norm(p1 + p3) > 1e-12 else p1
            return mid, d13 / 2.0
        else:
            mid = (p2 + p3) / np.linalg.norm(p2 + p3) if np.linalg.norm(p2 + p3) > 1e-12 else p2
            return mid, d23 / 2.0
    
    c = c / norm_c
    cos_r = np.clip(np.dot(c, p1), -1.0, 1.0)
    R = np.arccos(cos_r)
    return c, R


def _cell_inradius(points, i, k_neighbors=10):
    """
    Compute the inradius of the Voronoi cell for point i.
    
    The Voronoi cell of point i is the set of points closer to i than to any other point.
    Its boundary consists of arcs of great circles (bisectors between i and its neighbors).
    The cell vertices are the circumcenters of Delaunay triangles (triples of mutually adjacent points).
    
    For each triple of neighbors (j, k, l) that form a Delaunay triangle with i,
    the circumcenter of (i, j, k) or similar is a vertex of the Voronoi cell.
    The inradius is the maximum angular distance from i to any cell vertex.
    """
    n = len(points)
    # Compute distances from point i to all others
    dists_i = np.array([np.arccos(np.clip(np.dot(points[i], points[j]), -1, 1)) for j in range(n)])
    dists_i[i] = 0.0
    
    # Get k nearest neighbors
    neighbor_idx = np.argsort(dists_i)[1:k_neighbors+1]
    neighbors = [points[j] for j in neighbor_idx]
    
    # For each pair of neighbors, compute the circumcenter of (i, neighbor_a, neighbor_b)
    # This is a vertex of the Voronoi cell if it's within the cell
    inradii = []
    for a in range(len(neighbors)):
        for b in range(a+1, len(neighbors)):
            pa = neighbors[a]
            pb = neighbors[b]
            c, R = _spherical_circumcenter(points[i], pa, pb)
            # Check if c is actually in the Voronoi cell of i
            # c should be closer to i than to any other point
            dot_ic = np.clip(np.dot(c, points[i]), -1, 1)
            dist_ic = np.arccos(dot_ic)
            
            # Check that c is closer to i than to all other points
            is_in_cell = True
            for j in range(n):
                if j == i:
                    continue
                dot_cj = np.clip(np.dot(c, points[j]), -1, 1)
                dist_cj = np.arccos(dot_cj)
                if dist_cj < dist_ic - 1e-10:
                    is_in_cell = False
                    break
            
            if is_in_cell:
                inradii.append(dist_ic)
    
    if len(inradii) == 0:
        # Fallback: use half the distance to nearest neighbor
        return dists_i[neighbor_idx[0]] / 2.0
    
    return max(inradii)


def _all_inradii(points, k_neighbors=10):
    """Compute inradii for all points."""
    n = len(points)
    inradii = np.zeros(n)
    for i in range(n):
        inradii[i] = _cell_inradius(points, i, k_neighbors)
    return inradii


def _objective(x, k_neighbors=10):
    """
    Objective: sum of squared Voronoi inradii.
    """
    points = _normalize(x.reshape(N, 3))
    inradii = _all_inradii(points, k_neighbors)
    return np.sum(inradii ** 2)


def _covering_radius(points):
    """Compute the actual covering radius for evaluation."""
    points = _normalize(points)
    inradii = _all_inradii(points)
    return np.max(inradii) * 180.0 / np.pi


def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    Uses differentiable Voronoi inradius optimization.
    """
    start_time = time.time()
    max_time = 110  # Leave margin for safety
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    x0 = points.flatten().copy()
    best_x = x0.copy()
    best_obj = _objective(x0)
    
    # Phase 1: Coarse optimization with L-BFGS-B
    def obj_wrapper(x):
        return _objective(x)
    
    try:
        result = minimize(
            obj_wrapper,
            x0,
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-8}
        )
        if result.fun < best_obj:
            best_x = result.x
            best_obj = result.fun
    except Exception:
        pass
    
    if time.time() - start_time > max_time - 40:
        return _normalize(best_x.reshape(N, 3))
    
    # Phase 2: Local search with multiple restarts
    rng = np.random.RandomState(42)
    for restart in range(10):
        # Create perturbation
        scale = 0.2 * (1.0 - restart / 12.0)
        x_perturbed = best_x + rng.randn(N * 3) * scale
        try:
            result = minimize(
                obj_wrapper,
                x_perturbed,
                method='L-BFGS-B',
                options={'maxiter': 150, 'maxfun': 4000, 'ftol': 1e-8}
            )
            if result.fun < best_obj:
                best_x = result.x
                best_obj = result.fun
        except Exception:
            pass
        
        if time.time() - start_time > max_time - 20:
            break
    
    # Phase 3: Final fine-tuning with more iterations
    if time.time() - start_time < max_time - 5:
        try:
            result = minimize(
                obj_wrapper,
                best_x,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 8000, 'ftol': 1e-10}
            )
            if result.fun < best_obj:
                best_x = result.x
                best_obj = result.fun
        except Exception:
            pass
    
    points = _normalize(best_x.reshape(N, 3))
    
    # Validate: ensure points are on unit sphere and not degenerate
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms < 1e-6):
        # Fallback to Fibonacci
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.clip(1.0 - z * z, 0, 1))
        points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    return points

# EVOLVE-BLOCK-END
