import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

N = 14

# EVOLVE-BLOCK-START

def _min_triangle_area_fast(points):
    """
    Fast calculation of minimum triangle area using vectorized operations.
    Points: (N, 2) array. Returns minimum area (not doubled).
    """
    n = len(points)
    min_area = float('inf')
    for i in range(n - 2):
        pi = points[i]
        for j in range(i + 1, n - 1):
            v1 = points[j] - pi
            v2s = points[j + 1:] - pi
            cross_vals = v1[0] * v2s[:, 1] - v1[1] * v2s[:, 0]
            areas = 0.5 * np.abs(cross_vals)
            if len(areas) > 0:
                cur_min = np.min(areas)
                if cur_min < min_area:
                    min_area = cur_min
    return min_area

def _score(points):
    """
    Score = Min Triangle Area / Convex Hull Area.
    """
    try:
        if np.allclose(points, points[0]):
            return 0.0
        
        hull = ConvexHull(points)
        hull_area = hull.volume
        
        if hull_area < 1e-12:
            return 0.0
            
        min_tri_area = _min_triangle_area_fast(points)
        
        if min_tri_area < 1e-12:
            return 0.0
            
        return min_tri_area / hull_area
    except Exception:
        return 0.0

def _generate_ring_config(ring_counts, radii, angular_offsets):
    """
    Generate points from ring configuration.
    ring_counts: list of int, number of points in each ring
    radii: list of float, radius of each ring
    angular_offsets: list of float, rotation angle of each ring
    """
    points = []
    for k, count in enumerate(ring_counts):
        r = radii[k]
        offset = angular_offsets[k]
        if count == 0:
            continue
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + offset
        xs = r * np.cos(angles)
        ys = r * np.sin(angles)
        points.append(np.column_stack([xs, ys]))
    
    if not points:
        return np.zeros((N, 2))
        
    return np.vstack(points)

def _objective(params, ring_counts):
    """
    Objective function for minimization (returns negative score).
    params: [r1, r2, ..., ro1, ro2, ...]
    """
    n_rings = len(ring_counts)
    radii = params[:n_rings]
    offsets = params[n_rings:]
    
    # Ensure radii are positive
    radii = np.abs(radii)
    
    points = _generate_ring_config(ring_counts, radii, offsets)
    
    # Check for degenerate cases (e.g., points too close)
    # ConvexHull might fail if points are collinear or identical
    # _score handles exceptions, but we can add a penalty here for robustness
    score = _score(points)
    
    # Add small penalty if points are too close to each other (numerical stability)
    # Actually, _score returns 0 if min_tri_area is tiny, which is good.
    
    return -score

def _optimize_topology(ring_counts, time_budget):
    """
    Optimize a specific ring topology using Nelder-Mead.
    """
    n_rings = len(ring_counts)
    
    # Initial guess:
    # Radii: spread out from 0.2 to 1.0
    # Offsets: 0.0
    if n_rings == 1:
        # Single ring (center point + others? No, ring_counts[0] is count)
        # If ring_counts is [14], it's a single ring.
        # If ring_counts is [1, 13], it's center + 13.
        # Let's assume radii[0] is the first ring's radius.
        # For a single ring of 14, radius should be ~1.
        init_radii = np.ones(n_rings)
    else:
        # Spread radii
        init_radii = np.linspace(0.3, 1.0, n_rings)
        
    init_offsets = np.zeros(n_rings)
    x0 = np.concatenate([init_radii, init_offsets])
    
    # Bounds for Nelder-Mead (scipy minimize doesn't support bounds in Nelder-Mead directly 
    # unless we use 'L-BFGS-B' or similar, but Nelder-Mead is robust for small dims.
    # We will handle bounds by clipping inside the objective or just trusting NM for small dims.
    # Actually, let's use 'Nelder-Mead' as it is derivative-free and good for this.
    
    start_time = time.time()
    
    try:
        res = minimize(
            _objective, 
            x0, 
            args=(ring_counts,), 
            method='Nelder-Mead',
            options={
                'maxiter': 2000,
                'xatol': 1e-6,
                'fatol': 1e-8,
                'adaptive': True
            }
        )
        
        # Check time
        if time.time() - start_time > time_budget:
            return None, 0.0
            
        best_params = res.x
        best_score = -res.fun
        
        # Verify the score
        points = _generate_ring_config(ring_counts, np.abs(best_params[:n_rings]), best_params[n_rings:])
        verified_score = _score(points)
        
        return points, verified_score

    except Exception:
        return None, 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    best_points = None
    best_score = 0.0
    
    # Define topologies to explore
    # 1. Regular 14-gon (single ring)
    # 2. Center + 13 outer
    # 3. 2 inner (radius ~0.3) + 12 outer
    # 4. 3 inner + 11 outer
    # 5. 4 inner + 10 outer
    # 6. 5 inner + 9 outer
    # 7. 7 inner + 7 outer
    
    topologies = [
        [14],
        [1, 13],
        [2, 12],
        [3, 11],
        [4, 10],
        [5, 9],
        [7, 7],
        # Maybe some 3-ring topologies?
        [1, 6, 7],
        [2, 5, 7],
        [3, 5, 6],
    ]
    
    # Allocate time per topology
    # We have ~130s. 10 topologies. ~13s each.
    time_per_topology = 12.0
    
    for topo in topologies:
        if time.time() > deadline - 5:
            break
            
        current_time = time.time()
        remaining_for_this = min(time_per_topology, deadline - current_time - 2)
        
        if remaining_for_this < 1.0:
            continue
            
        points, score = _optimize_topology(topo, remaining_for_this)
        
        if points is not None and score > best_score:
            best_score = score
            best_points = points.copy()
            
    # Fallback if nothing worked or score is 0
    if best_points is None or best_score < 1e-6:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        best_score = _score(best_points)
        
    return best_points

# EVOLVE-BLOCK-END
