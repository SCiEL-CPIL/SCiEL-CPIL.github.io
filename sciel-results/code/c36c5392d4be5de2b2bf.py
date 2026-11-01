"""
Covering the sphere with 130 equal caps.

Strategy:
1. Start from a Fibonacci spiral (good initial distribution).
2. Use a geometric local search based on convex hull facets, which directly
   targets the covering radius definition.
3. For each facet (triangle) of the convex hull, compute the circumcenter.
   If the circumcenter is valid (on the correct side of the facet), move
   the three vertices toward it to reduce the circumradius.
4. If the circumcenter is invalid (obtuse triangle), move the two endpoints
   of the longest side toward their midpoint.
5. Iterate this process with adaptive step sizes.
6. Final polish with small perturbations and local optimization.

The key insight: the covering radius is determined by the largest empty cap,
which corresponds to the largest circumradius of the convex hull facets.
By directly optimizing the facets with the largest circumradii, we can
efficiently reduce the covering radius.
"""

import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

# Deadline guard
_START_TIME = time.time()
_TIME_LIMIT = 135  # seconds, leave 15s margin for the evaluator's 150s limit


def _elapsed():
    return time.time() - _START_TIME


def _time_exhausted(margin=5.0):
    return _elapsed() > _TIME_LIMIT - margin


def _normalize(v):
    """Normalize vectors to unit length, handling zero vectors."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return v / norms


def _covering_radius_from_hull(points):
    """
    Compute the covering radius (in radians) of a set of points on the unit sphere.
    
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest configuration point. This is equivalent to the largest circumradius
    of the convex hull facets.
    
    We compute this by finding the convex hull in 3D and computing the circumradius
    of each facet (triangle).
    """
    points = _normalize(points)
    
    # We need to compute the convex hull in 3D
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use a grid-based approximation
        return _covering_radius_grid(points)
    
    facets = hull.simplices
    max_radius = 0.0
    
    for tri in facets:
        a = points[tri[0]]
        b = points[tri[1]]
        c = points[tri[2]]
        
        # Find the circumcenter of triangle abc on the unit sphere
        # The circumcenter y satisfies:
        #   y . a = y . b = y . c  (equal angular distance to all three)
        #   |y| = 1
        #
        # From y.a = y.b: y . (a - b) = 0
        # From y.a = y.c: y . (a - c) = 0
        # So y is perpendicular to (a-b) and (a-c), i.e., y is parallel to (a-b) x (a-c)
        
        cross = np.cross(a - b, a - c)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            continue
        
        y = cross / norm_cross
        
        # Compute the angular distances from y to a, b, c
        d_a = np.arccos(np.clip(np.dot(y, a), -1, 1))
        d_b = np.arccos(np.clip(np.dot(y, b), -1, 1))
        d_c = np.arccos(np.clip(np.dot(y, c), -1, 1))
        
        # Check if y is on the correct side (same hemisphere as the triangle)
        normal = np.cross(b - a, c - a)
        normal_norm = np.linalg.norm(normal)
        if normal_norm < 1e-12:
            continue
        normal = normal / normal_norm
        
        # For a convex hull, the origin is inside, so the facet normal
        # points outward. The circumcenter should be on the same side as the normal.
        if np.dot(y, normal) < 0:
            # Obtuse triangle: the covering radius is half the longest side
            d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
            d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
            d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
            r = max(d_ab, d_bc, d_ca) / 2.0
        else:
            r = max(d_a, d_b, d_c)
        
        if r > max_radius:
            max_radius = r
    
    return max_radius


def _covering_radius_grid(points, n_grid=10000):
    """Fallback covering radius computation using a grid."""
    rng = np.random.default_rng(42)
    grid = rng.normal(size=(n_grid, 3))
    grid = _normalize(grid)
    
    dots = grid @ points.T
    max_dots = np.max(dots, axis=1)
    min_dists = np.arccos(np.clip(max_dots, -1, 1))
    
    return np.max(min_dists)


def _geometric_local_search(points, max_iter=5000, initial_step=0.02):
    """
    Perform geometric local search based on convex hull facets.
    
    For each facet with the largest circumradius, move the vertices toward
    the circumcenter (or midpoint of longest side for obtuse triangles).
    """
    step_size = initial_step
    no_improve_count = 0
    patience = 50
    
    best_points = points.copy()
    best_cr = _covering_radius_from_hull(points)
    
    for iteration in range(max_iter):
        if _time_exhausted(margin=10.0):
            break
        
        points = _normalize(points)
        
        try:
            hull = ConvexHull(points)
        except Exception:
            # If hull computation fails, break
            break
        
        facets = hull.simplices
        n_facets = len(facets)
        
        # Compute circumradius for each facet
        radii = np.zeros(n_facets)
        circumcenters = np.zeros((n_facets, 3))
        longest_side_midpoints = np.zeros((n_facets, 3))
        facet_types = np.zeros(n_facets, dtype=int)  # 0: acute, 1: obtuse
        
        for idx, tri in enumerate(facets):
            a = points[tri[0]]
            b = points[tri[1]]
            c = points[tri[2]]
            
            cross = np.cross(a - b, a - c)
            norm_cross = np.linalg.norm(cross)
            
            if norm_cross < 1e-12:
                radii[idx] = 0.0
                circumcenters[idx] = a
                longest_side_midpoints[idx] = a
                facet_types[idx] = 0
                continue
            
            y = cross / norm_cross
            
            d_a = np.arccos(np.clip(np.dot(y, a), -1, 1))
            d_b = np.arccos(np.clip(np.dot(y, b), -1, 1))
            d_c = np.arccos(np.clip(np.dot(y, c), -1, 1))
            
            normal = np.cross(b - a, c - a)
            normal_norm = np.linalg.norm(normal)
            if normal_norm < 1e-12:
                radii[idx] = 0.0
                circumcenters[idx] = a
                longest_side_midpoints[idx] = a
                facet_types[idx] = 0
                continue
            normal = normal / normal_norm
            
            if np.dot(y, normal) < 0:
                # Obtuse triangle
                d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
                d_bc = np.arcos(np.clip(np.dot(b, c), -1, 1))
                d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
                
                # Find the longest side
                if d_ab >= d_bc and d_ab >= d_ca:
                    r = d_ab / 2.0
                    midpoint = _normalize((a + b)[None, :])[0]
                elif d_bc >= d_ab and d_bc >= d_ca:
                    r = d_bc / 2.0
                    midpoint = _normalize((b + c)[None, :])[0]
                else:
                    r = d_ca / 2.0
                    midpoint = _normalize((c + a)[None, :])[0]
                
                radii[idx] = r
                circumcenters[idx] = y
                longest_side_midpoints[idx] = midpoint
                facet_types[idx] = 1
            else:
                # Acute triangle
                r = max(d_a, d_b, d_c)
                radii[idx] = r
                circumcenters[idx] = y
                longest_side_midpoints[idx] = y
                facet_types[idx] = 0
        
        # Find the facets with the largest circumradii
        # Sort facets by radius in descending order
        sorted_indices = np.argsort(-radii)
        
        # Get the top facets to optimize
        n_top = min(10, n_facets)
        top_indices = sorted_indices[:n_top]
        
        # Compute forces based on the top facets
        forces = np.zeros_like(points)
        
        for idx in top_indices:
            tri = facets[idx]
            r = radii[idx]
            
            if r < 1e-10:
                continue
            
            # Weight the force by the radius (larger facets get more attention)
            weight = r / (np.max(radii) + 1e-10)
            
            if facet_types[idx] == 0:
                # Acute triangle: move all three vertices toward circumcenter
                y = circumcenters[idx]
                for vertex_idx in tri:
                    p = points[vertex_idx]
                    # Direction from p to y, projected onto tangent space
                    direction = y - p
                    radial = np.dot(direction, p) * p
                    tangent = direction - radial
                    forces[vertex_idx] += tangent * weight
            else:
                # Obtuse triangle: move the two endpoints of the longest side toward midpoint
                midpoint = longest_side_midpoints[idx]
                # Need to identify which two vertices form the longest side
                a = points[tri[0]]
                b = points[tri[1]]
                c = points[tri[2]]
                
                d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
                d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
                d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
                
                if d_ab >= d_bc and d_ab >= d_ca:
                    # Longest side is a-b
                    for vertex_idx in [tri[0], tri[1]]:
                        p = points[vertex_idx]
                        direction = midpoint - p
                        radial = np.dot(direction, p) * p
                        tangent = direction - radial
                        forces[vertex_idx] += tangent * weight
                elif d_bc >= d_ab and d_bc >= d_ca:
                    # Longest side is b-c
                    for vertex_idx in [tri[1], tri[2]]:
                        p = points[vertex_idx]
                        direction = midpoint - p
                        radial = np.dot(direction, p) * p
                        tangent = direction - radial
                        forces[vertex_idx] += tangent * weight
                else:
                    # Longest side is c-a
                    for vertex_idx in [tri[2], tri[0]]:
                        p = points[vertex_idx]
                        direction = midpoint - p
                        radial = np.dot(direction, p) * p
                        tangent = direction - radial
                        forces[vertex_idx] += tangent * weight
        
        # Normalize forces
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms = np.where(force_norms < 1e-12, 1.0, force_norms)
        forces = forces / force_norms
        
        # Update points
        points_new = points + step_size * forces
        points_new = _normalize(points_new)
        
        # Check if the covering radius improved
        new_cr = _covering_radius_from_hull(points_new)
        
        if new_cr < best_cr - 1e-10:
            best_cr = new_cr
            best_points = points_new.copy()
            points = points_new
            no_improve_count = 0
        else:
            points = points_new  # Still accept for exploration
            no_improve_count += 1
            if no_improve_count >= patience:
                step_size *= 0.5
                no_improve_count = 0
                if step_size < 1e-8:
                    break
    
    return best_points, best_cr


def _final_polish(points):
    """
    Final polish with small random perturbations and local optimization.
    """
    best_points = points.copy()
    best_cr = _covering_radius_from_hull(points)
    
    rng = np.random.default_rng(12345)
    
    for _ in range(200):
        if _time_exhausted(margin=10.0):
            break
        
        # Try small random perturbations
        perturbation = rng.normal(size=(N, 3)) * 0.005
        points_perturbed = _normalize(best_points + perturbation)
        
        cr_perturbed = _covering_radius_from_hull(points_perturbed)
        
        if cr_perturbed < best_cr:
            best_cr = cr_perturbed
            best_points = points_perturbed
    
    # Run geometric local search again with smaller steps
    if not _time_exhausted(margin=20.0):
        best_points, best_cr = _geometric_local_search(best_points, max_iter=2000, initial_step=0.005)
    
    return best_points, best_cr


def construct():
    """
    Construct 130 points on the unit sphere minimizing the covering radius.
    
    Returns an array of shape (130, 3) with unit vectors.
    """
    global _START_TIME
    _START_TIME = time.time()
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points = _normalize(points)
    
    # Geometric local search
    points, cr = _geometric_local_search(points, max_iter=5000, initial_step=0.02)
    
    # Final polish
    if not _time_exhausted(margin=20.0):
        points, cr = _final_polish(points)
    
    # Ensure the result is valid
    points = _normalize(points)
    
    # Check for coincident points
    dots = points @ points.T
    np.fill_diagonal(dots, 0.0)
    if np.max(dots) > 0.9999:
        # Perturb coincident points
        rng = np.random.default_rng(0)
        for i in range(N):
            for j in range(i+1, N):
                if np.dot(points[i], points[j]) > 0.9999:
                    perturbation = rng.normal(size=3) * 0.01
                    points[i] = _normalize(points[i] + perturbation)[0]
    
    return points
