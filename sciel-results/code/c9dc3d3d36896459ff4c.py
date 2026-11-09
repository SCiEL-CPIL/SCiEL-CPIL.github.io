import numpy as np
import time
from scipy.spatial import ConvexHull
import sys

N = 50

def construct():
    # Start time
    start_time = time.time()
    timeout = 145.0  # Safety margin for 150s limit
    
    # 1. Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    P = P / np.linalg.norm(P, axis=1, keepdims=True)
    
    best_P = P.copy()
    best_radius = np.inf
    
    # Helper to calculate covering radius
    def calc_covering_radius(points):
        try:
            hull = ConvexHull(points)
            # hull.simplices gives the indices of the points forming each facet
            # For 3D points, facets are triangles
            max_rad = 0.0
            for simplex in hull.simplices:
                i, j, k = simplex
                u = points[i]
                v = points[j]
                w = points[k]
                
                # Compute circumcenter of the spherical triangle
                # Normal to the plane containing u, v, w
                n = np.cross(v - u, w - u)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-10:
                    # Degenerate triangle
                    continue
                c = n / n_norm
                
                # Ensure c is on the same side as the origin? 
                # The circumcenter on the sphere is c or -c.
                # The facet is part of the convex hull, so the origin is on one side.
                # The circumcenter of the triangle on the sphere is the point c such that c is equidistant from u, v, w.
                # c = n / ||n|| is the normal.
                # We need to check if c is the correct pole.
                # The distance is arccos(c dot u).
                # If c dot u is negative, the angle is > 90, which might be the "back" side.
                # But for a convex hull of points on the unit sphere, the facet normal points OUTWARD.
                # The circumcenter of the triangle on the sphere is the intersection of the sphere with the line through the origin perpendicular to the plane?
                # Yes.
                # Let's check the dot product.
                dot = np.dot(c, u)
                # The angle is arccos(dot).
                # If dot is negative, the angle is large.
                # Is it possible that -c is the circumcenter?
                # The circumcenter is unique.
                # For a triangle on the sphere, the circumcenter is the point c such that c.u = c.v = c.w.
                # c = n/||n|| satisfies this.
                rad = np.arccos(np.clip(dot, -1.0, 1.0))
                if rad > max_rad:
                    max_rad = rad
            return max_rad
        except Exception:
            return np.pi # Invalid hull, return large radius
    
    # Initial evaluation
    current_radius = calc_covering_radius(P)
    if current_radius < best_radius:
        best_radius = current_radius
        best_P = P.copy()
    
    # Optimization Loop
    # We use a gradient-like step towards circumcenters of the worst facets
    step_size = 0.1
    decay = 0.995
    min_step = 1e-6
    
    while time.time() - start_time < timeout:
        try:
            hull = ConvexHull(P)
            simplices = hull.simplices
            
            # Find the maximum circumradius and the active facets
            max_rad = 0.0
            active_facets = []
            circumcenters = {}
            
            for simplex in simplices:
                i, j, k = simplex
                u = P[i]
                v = P[j]
                w = P[k]
                
                n = np.cross(v - u, w - u)
                n_norm = np.linalg.norm(n)
                if n_norm < 1e-10:
                    continue
                c = n / n_norm
                dot = np.dot(c, u)
                rad = np.arccos(np.clip(dot, -1.0, 1.0))
                
                circumcenters[simplex] = (c, rad)
                if rad > max_rad:
                    max_rad = rad
                    active_facets = [simplex]
                elif np.isclose(rad, max_rad, rtol=1e-5):
                    active_facets.append(simplex)
            
            if not active_facets:
                break
                
            # Update points
            # We want to move points i, j, k towards c
            # To avoid conflicting updates, we accumulate deltas
            deltas = np.zeros_like(P)
            
            for simplex in active_facets:
                c, rad = circumcenters[simplex]
                i, j, k = simplex
                # Move towards circumcenter
                # The direction is c - P[i]
                # We project this onto the tangent space? 
                # Or just update and re-normalize?
                # Updating and re-normalizing is simpler and works well.
                
                deltas[i] += c - P[i]
                deltas[j] += c - P[j]
                deltas[k] += c - P[k]
            
            # Apply step
            # Normalize deltas to prevent huge jumps if many facets are active
            delta_norm = np.linalg.norm(deltas)
            if delta_norm > 0:
                deltas = deltas / delta_norm * step_size
                P = P + deltas
                # Normalize P back to unit sphere
                P = P / np.linalg.norm(P, axis=1, keepdims=True)
            
            # Evaluate new radius
            new_radius = calc_covering_radius(P)
            
            if new_radius < best_radius:
                best_radius = new_radius
                best_P = P.copy()
            
            # Decay step size
            step_size *= decay
            if step_size < min_step:
                # If step size is too small, we might be stuck in a local minimum
                # We can try to break out by adding noise
                # Or just stop if we haven't improved in a while
                # For now, we'll just stop if step is tiny
                pass
                
        except Exception as e:
            # If hull computation fails, break
            break
            
    return best_P
