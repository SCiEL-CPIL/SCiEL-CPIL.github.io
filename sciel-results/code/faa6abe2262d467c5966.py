"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import time
import itertools

def _min_triangle_area(points):
    """
    Calculate the minimum area of any triangle formed by 3 points in the set.
    Points: (N, 2) array.
    """
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        pi = points[i]
        for j in range(i + 1, n):
            pj = points[j]
            if j + 1 < n:
                pk = points[j + 1:]
                v1 = pj - pi
                v2 = pk - pi
                cross = v1[0] * v2[:, 1] - v1[1] * v2[:, 0]
                areas = 0.5 * np.abs(cross)
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
    return min_area

def _hull_area(points):
    """
    Calculate the area of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    """
    min_area = _min_triangle_area(points)
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    return min_area / hull_area

def _points_from_affine(params):
    """
    Generate points from an affine transformation of a regular polygon.
    params: [a, b, c, d, tx, ty] where M = [[a, b], [c, d]] and t = (tx, ty)
    The base points are vertices of a regular N-gon on the unit circle.
    """
    a, b, c, d, tx, ty = params
    base_angles = 2 * np.pi * np.arange(N) / N
    base_points = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    M = np.array([[a, b], [c, d]])
    points = base_points @ M.T + np.array([tx, ty])
    return points

def _objective_affine(params):
    """
    Objective function for affine transformation optimization.
    Returns negative score (to be minimized).
    """
    points = _points_from_affine(params)
    return -_score(points)

def _points_from_rings(ring_params):
    """
    Generate points from a concentric ring structure.
    ring_params: list of (radius, angle_offset, num_points) for each ring.
    """
    points = []
    for radius, angle_offset, num_points in ring_params:
        if num_points > 0:
            angles = 2 * np.pi * np.arange(num_points) / num_points + angle_offset
            ring_points = np.column_stack([
                radius * np.cos(angles),
                radius * np.sin(angles)
            ])
            points.append(ring_points)
    if not points:
        return np.zeros((N, 2))
    return np.vstack(points)

def _score_from_ring_params(ring_params):
    """
    Calculate score for given ring parameters.
    ring_params: flat array [r0, a0, r1, a1, ...]
    """
    points = []
    idx = 0
    for num_points in ring_sizes:
        if num_points > 0:
            r = ring_params[idx]
            a = ring_params[idx + 1]
            idx += 2
            angles = 2 * np.pi * np.arange(num_points) / num_points + a
            ring_points = np.column_stack([
                r * np.cos(angles),
                r * np.sin(angles)
            ])
            points.append(ring_points)
    if not points:
        return 0.0
    pts = np.vstack(points)
    if len(pts) != N:
        return 0.0
    return _score(pts)

def _objective_ring_params(ring_params):
    """
    Objective function for ring parameter optimization.
    """
    return -_score_from_ring_params(ring_params)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    best_score = 0.0
    best_points = None
    
    # Strategy 1: Affine transformation of regular polygon
    # 6 parameters: a, b, c, d, tx, ty
    # Regular polygon already has tx=ty=0, and M=I gives unit circle
    # We want to explore non-trivial affine transformations
    
    # Bounds for affine parameters
    # Keep the transformation bounded to avoid degenerate cases
    bounds_affine = [(-2, 2), (-2, 2), (-2, 2), (-2, 2), (-1, 1), (-1, 1)]
    
    # Initial guess: identity transformation (regular polygon on unit circle)
    x0_affine = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    
    # Try differential evolution for global search
    if time.time() < deadline - 30:
        try:
            de_result = differential_evolution(
                _objective_affine, bounds_affine, 
                seed=42, maxiter=100, popsize=15,
                tol=1e-6, polish=True
            )
            score_de = -de_result.fun
            points_de = _points_from_affine(de_result.x)
            
            if score_de > best_score:
                best_score = score_de
                best_points = points_de.copy()
        except Exception:
            pass
    
    # Try Nelder-Mead from the DE result
    if best_points is not None and time.time() < deadline - 15:
        try:
            nm_result = minimize(
                _objective_affine, de_result.x, method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10}
            )
            score_nm = -nm_result.fun
            points_nm = _points_from_affine(nm_result.x)
            
            if score_nm > best_score:
                best_score = score_nm
                best_points = points_nm.copy()
        except Exception:
            pass
    
    # Strategy 2: Concentric rings
    # Try different ring configurations
    ring_configs = [
        # (num_points_per_ring)
        (14,),           # Single ring (regular polygon)
        (1, 13),         # Center + outer ring
        (2, 12),         # Inner pair + outer ring
        (3, 11),         # Inner triangle + outer ring
        (4, 10),         # Inner square + outer ring
        (5, 9),          # Inner pentagon + outer ring
        (7, 7),          # Two rings of 7
        (2, 2, 10),      # Two inner pairs + outer ring
        (3, 3, 8),       # Two inner triangles + outer ring
        (1, 1, 12),      # Two centers + outer ring
    ]
    
    for config in ring_configs:
        if time.time() > deadline - 20:
            break
        
        ring_sizes = list(config)
        num_rings = len(ring_sizes)
        
        # Parameters: for each ring, [radius, angle_offset]
        # Radius bounds: inner rings smaller, outer ring larger
        bounds_ring = []
        x0_ring = []
        
        for i, ns in enumerate(ring_sizes):
            if i == num_rings - 1:
                # Outer ring: radius around 1.0
                bounds_ring.extend([(0.5, 2.0), (-np.pi, np.pi)])
                x0_ring.extend([1.0, 0.0])
            else:
                # Inner rings: smaller radii
                r_max = 0.8 * (i + 1) / num_rings
                bounds_ring.extend([(0.0, r_max), (-np.pi, np.pi)])
                x0_ring.extend([r_max * 0.5, 0.0])
        
        # Optimize ring parameters
        try:
            de_result_ring = differential_evolution(
                _objective_ring_params, bounds_ring,
                seed=42, maxiter=50, popsize=12,
                tol=1e-5, polish=True
            )
            score_ring = -de_result_ring.fun
            
            if score_ring > best_score:
                best_score = score_ring
                best_points = _points_from_ring_params(de_result_ring.x).copy()
        except Exception:
            pass
        
        # Refine with Nelder-Mead
        if time.time() < deadline - 10:
            try:
                nm_result_ring = minimize(
                    _objective_ring_params, de_result_ring.x, method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                score_ring_nm = -nm_result_ring.fun
                
                if score_ring_nm > best_score:
                    best_score = score_ring_nm
                    best_points = _points_from_ring_params(nm_result_ring.x).copy()
            except Exception:
                pass
    
    # Fallback to regular polygon if nothing else worked
    if best_points is None:
        base_angles = 2 * np.pi * np.arange(N) / N
        best_points = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    return best_points
# EVOLVE-BLOCK-END
