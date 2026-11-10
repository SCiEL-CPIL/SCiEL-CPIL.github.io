import numpy as np
import time
from scipy.spatial import ConvexHull
import math

N = 130

def construct():
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    start_time = time.time()
    deadline = 105.0 # Safety margin
    
    # Precompute some constants
    step_size = 0.05
    
    while time.time() - start_time < deadline:
        # Compute Convex Hull
        try:
            hull = ConvexHull(P)
            facets = hull.simplices
        except Exception:
            break
            
        # Calculate covering radius for each facet
        # Facet is a triangle (u, v, w)
        max_radius = 0.0
        worst_facet_idx = -1
        worst_point = None
        
        radii = np.zeros(len(facets))
        worst_points = np.zeros((len(facets), 3))
        
        for k, facet in enumerate(facets):
            u = P[facet[0]]
            v = P[facet[1]]
            w = P[facet[2]]
            
            # Spherical triangle circumradius
            # Side lengths
            a = np.arccos(np.clip(np.dot(v, w), -1.0, 1.0))
            b = np.arccos(np.clip(np.dot(u, w), -1.0, 1.0))
            c = np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))
            
            # Check if obtuse
            # If a^2 > b^2 + c^2, etc.
            # For spherical triangles, the condition is slightly different but for small angles it's similar.
            # Let's just compute the circumcenter and check if it's inside.
            # Circumcenter C satisfies C.u = C.v = C.w
            # C is proportional to (v x w) + (w x u) + (u x v) ? No, that's the centroid.
            # The circumcenter is the normalized vector of the solution to the linear system.
            # Let C = alpha * u + beta * v + gamma * w ? No.
            
            # A robust way to find the circumcenter of a spherical triangle:
            # The circumcenter is the intersection of the great circles that are the perpendicular bisectors.
            # The normal to the bisector of arc vw is (v+w) x (v-w)? No.
            # The bisector plane contains C and the origin, and is perpendicular to the chord vw.
            # So C is orthogonal to (v-w) and (v+w)? No.
            # C . (v-w) = 0 => C.v = C.w.
            # So C is in the null space of the matrix with rows (v-w), (w-u), (u-v)?
            # Actually, C is the normalized cross product of (v-w) and (w-u)?
            # Let n1 = v - w
            # Let n2 = w - u
            # C is perpendicular to n1 and n2? No, C.v = C.w => C.(v-w)=0.
            # So C is perpendicular to v-w and w-u.
            # So C is parallel to (v-w) x (w-u).
            
            n1 = v - w
            n2 = w - u
            C = np.cross(n1, n2)
            
            # Normalize C
            norm_C = np.linalg.norm(C)
            if norm_C < 1e-8:
                # Degenerate
                radii[k] = 0
                worst_points[k] = u
                continue
            C = C / norm_C
            
            # Check if C is inside the triangle
            # C is inside if it has the same orientation as the triangle vertices relative to the normal?
            # Or simply, check if the angles are less than pi?
            # A simpler check: The circumcenter is inside if the triangle is acute.
            # We can check if the dot product of C with the normal of the triangle is positive?
            # The normal of the triangle (u,v,w) is n = (v-u) x (w-u).
            # If C . n > 0, it's on the same side.
            n_tri = np.cross(v-u, w-u)
            if np.dot(C, n_tri) < 0:
                C = -C
                
            # Now C is the circumcenter.
            # The radius is the angle between C and u.
            R = np.arccos(np.clip(np.dot(C, u), -1.0, 1.0))
            
            # Check if the circumcenter is actually the farthest point.
            # If the triangle is obtuse, the farthest point is the midpoint of the longest side.
            # We can check if R is greater than the half-sides?
            # If the triangle is obtuse, the circumcenter is outside, and the "radius" calculated above might be the distance to the "other side" of the sphere?
            # Actually, if the circumcenter is outside the triangle, the maximum distance to the vertices within the triangle is at the midpoint of the longest edge.
            # Let's check the angles.
            # If any angle > pi/2, it's obtuse.
            # Angle at u: cos(A) = (b^2 + c^2 - a^2) / (2bc)? Spherical law of cosines.
            # cos(a) = cos(b)cos(c) + sin(b)sin(c)cos(A)
            # cos(A) = (cos(a) - cos(b)cos(c)) / (sin(b)sin(c))
            
            cos_A = (np.cos(a) - np.cos(b)*np.cos(c)) / (np.sin(b)*np.sin(c) + 1e-12)
            cos_B = (np.cos(b) - np.cos(c)*np.cos(a)) / (np.sin(c)*np.sin(a) + 1e-12)
            cos_C_ = (np.cos(c) - np.cos(a)*np.cos(b)) / (np.sin(a)*np.sin(b) + 1e-12)
            
            is_obtuse = (cos_A < 0) or (cos_B < 0) or (cos_C_ < 0)
            
            if is_obtuse:
                # Find longest side
                sides = [a, b, c]
                longest_idx = np.argmax(sides)
                if longest_idx == 0: # a is longest (vw)
                    mid = (v + w) / np.linalg.norm(v + w)
                    R = np.arccos(np.clip(np.dot(mid, v), -1.0, 1.0)) # Distance from mid to v
                    worst_point = mid
                elif longest_idx == 1: # b is longest (uw)
                    mid = (u + w) / np.linalg.norm(u + w)
                    R = np.arccos(np.clip(np.dot(mid, u), -1.0, 1.0))
                    worst_point = mid
                else: # c is longest (uv)
                    mid = (u + v) / np.linalg.norm(u + v)
                    R = np.arccos(np.clip(np.dot(mid, u), -1.0, 1.0))
                    worst_point = mid
            else:
                worst_point = C
                
            radii[k] = R
            worst_points[k] = worst_point
            
        # Find the worst facet
        max_idx = np.argmax(radii)
        max_radius = radii[max_idx]
        
        if max_radius < 0.1: # Very good, stop?
            break
            
        # Move the vertices of the worst facet away from the worst point
        facet = facets[max_idx]
        wp = worst_points[max_idx]
        
        # Calculate direction to move each vertex
        # Move away from wp in the tangent plane
        for idx in facet:
            p = P[idx]
            # Vector from wp to p
            v = p - wp
            # Project v onto tangent plane at p
            # Tangent plane is orthogonal to p
            v_tan = v - np.dot(v, p) * p
            # Normalize
            norm_v = np.linalg.norm(v_tan)
            if norm_v > 1e-8:
                v_tan = v_tan / norm_v
                P[idx] += step_size * v_tan
                
        # Normalize all points
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        
    return P
