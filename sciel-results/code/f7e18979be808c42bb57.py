"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
import sys

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    # 1. Initialization: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = np.inf
    
    def compute_covering_radius(pts):
        """Compute covering radius using convex hull facets."""
        try:
            hull = ConvexHull(pts)
            facets = hull.simplices
            max_r = 0.0
            for f in facets:
                # Get vertices
                v1, v2, v3 = pts[f[0]], pts[f[1]], pts[f[2]]
                # Compute circumcenter of the spherical triangle
                # The circumcenter is the intersection of the plane spanned by v1, v2, v3 
                # with the line perpendicular to the plane passing through the origin? 
                # No, the circumcenter of a spherical triangle lies on the sphere.
                # It is the point equidistant from v1, v2, v3.
                # Let o = a*v1 + b*v2 + c*v3.
                # o.v1 = o.v2 = o.v3 = cos(R)
                # This is a linear system.
                # [v1.v1 v1.v2 v1.v3] [a]   [cosR]
                # [v2.v1 v2.v2 v2.v3] [b] = [cosR]
                # [v3.v1 v3.v2 v3.v3] [c]   [cosR]
                # But we don't know cosR.
                # Alternative: The circumcenter is the normalized cross product of the edges? No.
                # The circumcenter is the pole of the plane passing through the midpoints?
                
                # Simple method: Solve for o such that |o|^2=1 and o.v1=o.v2=o.v3.
                # This implies o is in the span of v1, v2, v3.
                # Let o = x*v1 + y*v2 + z*v3.
                # We want o to be equidistant from v1, v2, v3.
                # This is equivalent to o being the circumcenter of the Euclidean triangle 
                # formed by projecting v1, v2, v3 onto a tangent plane? 
                # Approximation for small triangles:
                # Use the formula for circumcenter of a triangle in 3D.
                # For points on the unit sphere, the circumcenter o can be found by:
                # o = (v1 + v2 + v3) / ||v1 + v2 + v3|| ? No, that's the centroid.
                
                # Correct method:
                # The circumcenter o satisfies (o - v1)^2 = (o - v2)^2 = (o - v3)^2 = 2(1-cosR)
                # o.o - 2o.v1 + v1.v1 = o.o - 2o.v2 + v2.v2
                # 2o.(v2-v1) = 0 => o is perpendicular to (v2-v1).
                # So o is in the intersection of the planes perpendicular to (v2-v1) and (v3-v1) passing through the origin?
                # No, the plane perpendicular to (v2-v1) passing through the midpoint?
                # The set of points equidistant from v1 and v2 is the plane passing through the origin and perpendicular to (v2-v1).
                # So o is the intersection of the sphere with the line defined by the cross product of the normals?
                # Normal n1 = v2 - v1. Plane 1: x . n1 = 0.
                # Normal n2 = v3 - v1. Plane 2: x . n2 = 0.
                # Intersection is the line along n1 x n2.
                # So o is parallel to (v2-v1) x (v3-v1).
                # Let o = normalize( (v2-v1) x (v3-v1) ).
                # Check: Is this the circumcenter?
                # Yes, for a spherical triangle, the circumcenter is the pole of the plane defined by the vertices?
                # No, the pole is equidistant from the vertices only if the triangle is "right" or something?
                # Actually, the point o = normalize( (v2-v1) x (v3-v1) ) is equidistant from v1, v2, v3.
                # Proof: o . (v2-v1) = 0 and o . (v3-v1) = 0.
                # So o.v2 = o.v1 and o.v3 = o.v1.
                # So o is equidistant.
                
                edge1 = v2 - v1
                edge2 = v3 - v1
                normal = np.cross(edge1, edge2)
                if np.linalg.norm(normal) < 1e-10:
                    # Degenerate
                    continue
                o = normal / np.linalg.norm(normal)
                
                # Calculate radius
                # cos(R) = o . v1
                cos_r = np.dot(o, v1)
                # Clip for numerical stability
                cos_r = np.clip(cos_r, -1.0, 1.0)
                r_val = np.arccos(cos_r)
                
                if r_val > max_r:
                    max_r = r_val
            return max_r
        except Exception:
            return np.inf

    current_radius = compute_covering_radius(points)
    if current_radius < best_radius:
        best_radius = current_radius
        best_points = points.copy()

    # 2. Optimization: Gradient Descent on Facet Circumradii
    # We use a heuristic: For each facet, compute circumcenter o and radius R.
    # Apply force to vertices to move them towards o (shrinking the triangle).
    # Weight by R^2 to focus on the worst facets.
    
    step_size = 0.01
    decay = 0.999
    iterations = 0
    
    while time.time() < deadline:
        try:
            hull = ConvexHull(points)
            facets = hull.simplices
            
            forces = np.zeros_like(points)
            
            # Compute circumcenters and radii for all facets
            # To save time, we might only process the worst ones, but N=130 is small.
            # Facets count is ~ 2N - 4 = 256.
            
            # First pass: find max radius to normalize weights?
            # Or just use R^2 directly.
            
            for f in facets:
                v1, v2, v3 = points[f[0]], points[f[1]], points[f[2]]
                
                edge1 = v2 - v1
                edge2 = v3 - v1
                normal = np.cross(edge1, edge2)
                norm_n = np.linalg.norm(normal)
                if norm_n < 1e-10:
                    continue
                
                o = normal / norm_n
                cos_r = np.dot(o, v1)
                cos_r = np.clip(cos_r, -1.0, 1.0)
                r_val = np.arccos(cos_r)
                
                # Weight: larger radius -> stronger force
                w = r_val * r_val
                
                # Force: move vertices towards circumcenter
                # F_v = w * (o - v)
                # But we must ensure the force is tangent to the sphere.
                # (o - v) is already tangent? 
                # o.v = cos(R). v.v = 1.
                # (o-v).v = o.v - 1 = cos(R) - 1 != 0.
                # So we need to project (o-v) onto the tangent plane at v.
                # Tangent plane at v: x . v = 0.
                # Projection of u onto tangent plane: u - (u.v)v.
                
                for idx in [0, 1, 2]:
                    v = points[f[idx]]
                    diff = o - v
                    # Project diff to tangent plane
                    dot = np.dot(diff, v)
                    tangent_force = diff - dot * v
                    forces[f[idx]] += w * tangent_force
            
            # Update points
            # Normalize forces to avoid huge steps
            force_norm = np.linalg.norm(forces, axis=1)
            max_force = np.max(force_norm) if len(force_norm) > 0 else 1.0
            if max_force > 0:
                forces = forces / max_force
            
            points = points + step_size * forces
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
            
            # Evaluate
            current_radius = compute_covering_radius(points)
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = points.copy()
            
            # Adaptive step size
            # If radius increased, reduce step size
            # We don't have the previous radius stored easily in this loop structure, 
            # but we can compare with best_radius? No, best_radius is global min.
            # Let's just decay step size slowly.
            step_size *= decay
            iterations += 1
            
        except Exception as e:
            # If hull fails, break
            break

    # Final check: ensure best_points is valid
    # If best_points is still inf, return initial
    if not np.isfinite(best_radius):
        return points
    
    return best_points

# EVOLVE-BLOCK-END
