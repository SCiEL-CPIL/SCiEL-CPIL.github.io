"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

def _normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def _angular_distance(p, q):
    dot = np.clip(np.dot(p, q), -1.0, 1.0)
    return np.arccos(dot)

def _circumcenter_sphere_vec(points, simplices):
    """
    Vectorized calculation of circumcenters for a set of simplices.
    Returns array of circumcenters and radii.
    """
    p1 = points[simplices[:, 0]]
    p2 = points[simplices[:, 1]]
    p3 = points[simplices[:, 2]]
    
    # Side lengths
    a = _angular_distance(p2, p3)
    b = _angular_distance(p1, p3)
    c = _angular_distance(p1, p2)
    
    cos_a = np.cos(a)
    cos_b = np.cos(b)
    cos_c = np.cos(c)
    sin_b = np.sin(b)
    sin_c = np.sin(c)
    sin_a = np.sin(a)
    
    # Handle degeneracies
    valid = (sin_b * sin_c > 1e-10) & (sin_a * sin_c > 1e-10) & (sin_a * sin_b > 1e-10)
    
    cos_A = np.zeros(len(p1))
    cos_B = np.zeros(len(p1))
    cos_C = np.zeros(len(p1))
    
    cos_A[valid] = (cos_a[valid] - cos_b[valid]*cos_c[valid]) / (sin_b[valid]*sin_c[valid])
    cos_B[valid] = (cos_b[valid] - cos_a[valid]*cos_c[valid]) / (sin_a[valid]*sin_c[valid])
    cos_C[valid] = (cos_c[valid] - cos_a[valid]*cos_b[valid]) / (sin_a[valid]*sin_b[valid])
    
    cos_A = np.clip(cos_A, -1.0, 1.0)
    cos_B = np.clip(cos_B, -1.0, 1.0)
    cos_C = np.clip(cos_C, -1.0, 1.0)
    
    A = np.arccos(cos_A)
    B = np.arccos(cos_B)
    C = np.arccos(cos_C)
    
    sin_A = np.sin(A)
    sin_B = np.sin(B)
    sin_C = np.sin(C)
    
    c_vec = sin_A[:, None] * p1 + sin_B[:, None] * p2 + sin_C[:, None] * p3
    c_norm = np.linalg.norm(c_vec, axis=1)
    
    # Handle zero norms
    c_norm_safe = np.where(c_norm < 1e-10, 1.0, c_norm)
    c_unit = c_vec / c_norm_safe[:, None]
    
    # Radius
    dot_p1 = np.sum(c_unit * p1, axis=1)
    dot_p1 = np.clip(dot_p1, -1.0, 1.0)
    radii = np.arccos(dot_p1)
    
    # For invalid triangles, set radius to infinity or large value
    radii[~valid] = np.pi
    
    return c_unit, radii

def _is_point_in_spherical_triangle_vec(p, p1, p2, p3):
    """
    Vectorized check if points p are inside triangles (p1, p2, p3).
    """
    n12 = np.cross(p1, p2)
    n23 = np.cross(p2, p3)
    n31 = np.cross(p3, p1)
    
    # Normalize normals
    n12_norm = np.linalg.norm(n12, axis=1)
    n23_norm = np.linalg.norm(n23, axis=1)
    n31_norm = np.linalg.norm(n31, axis=1)
    
    n12_norm_safe = np.where(n12_norm < 1e-10, 1.0, n12_norm)
    n23_norm_safe = np.where(n23_norm < 1e-10, 1.0, n23_norm)
    n31_norm_safe = np.where(n31_norm < 1e-10, 1.0, n31_norm)
    
    n12 = n12 / n12_norm_safe[:, None]
    n23 = n23 / n23_norm_safe[:, None]
    n31 = n31 / n31_norm_safe[:, None]
    
    # Check side for edge p1-p2 (normal n12)
    s_test_12 = np.sum(p * n12, axis=1)
    s_ref_12 = np.sum(p3 * n12, axis=1)
    ok_12 = (s_test_12 * s_ref_12) >= 0
    
    # Check side for edge p2-p3 (normal n23)
    s_test_23 = np.sum(p * n23, axis=1)
    s_ref_23 = np.sum(p1 * n23, axis=1)
    ok_23 = (s_test_23 * s_ref_23) >= 0
    
    # Check side for edge p3-p1 (normal n31)
    s_test_31 = np.sum(p * n31, axis=1)
    s_ref_31 = np.sum(p2 * n31, axis=1)
    ok_31 = (s_test_31 * s_ref_31) >= 0
    
    return ok_12 & ok_23 & ok_31

def _covering_radius_and_gradients(points):
    """
    Compute covering radius and identify worst facets.
    Returns: max_radius, list of (i, j, k, circumcenter, is_inside) for top facets.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi, []
    
    simplices = hull.simplices
    c_unit, radii = _circumcenter_sphere_vec(points, simplices)
    
    # Check if circumcenters are inside the triangles
    p1 = points[simplices[:, 0]]
    p2 = points[simplices[:, 1]]
    p3 = points[simplices[:, 2]]
    
    inside = _is_point_in_spherical_triangle_vec(c_unit, p1, p2, p3)
    
    # If circumcenter is outside, the effective radius for that facet is 
    # the max distance to the vertices (which is half the longest side? No, it's the distance to the farthest vertex)
    # Actually, if the circumcenter is outside, the largest empty circle in the triangle 
    # has its center at the midpoint of the longest side? 
    # Wait, for covering radius, we care about the point on the sphere farthest from any vertex.
    # In a spherical triangle, the farthest point from the vertices is either the circumcenter 
    # (if inside) or one of the vertices? No, the vertices are distance 0.
    # The farthest point in the *interior* is the circumcenter if it's inside.
    # If the circumcenter is outside, the maximum distance to a vertex within the triangle 
    # is achieved at one of the vertices? No, the distance function is convex.
    # The maximum of a convex function on a convex set is at the boundary.
    # The boundary of the spherical triangle consists of the three arcs.
    # On an arc between p1 and p2, the distance to p1 is 0 at p1, increases to d12/2 at midpoint, then decreases?
    # No, distance to nearest vertex.
    # If circumcenter is outside, the triangle is "obtuse". The farthest point from the set {p1, p2, p3} 
    # within the triangle is the midpoint of the longest side?
    # Let's assume for now that if outside, we use the max vertex distance? 
    # No, that's 0.
    # The covering radius is the max over the *sphere*.
    # The sphere is covered by the Voronoi cells. The worst point in a Voronoi cell is the circumcenter of the corresponding facet.
    # If the circumcenter is outside the facet, it is NOT in the Voronoi cell.
    # In that case, the worst point in the Voronoi cell is one of the vertices of the facet?
    # No, the Voronoi cell is the set of points closer to pi than to any other pj.
    # The boundary of the Voronoi cell for pi is the set of points equidistant to pi and some pj.
    # The "worst" point for the *global* covering is the point on the sphere that is farthest from ALL points.
    # This point must be a vertex of the Delaunay triangulation's Voronoi diagram?
    # Yes, the covering radius is the maximum over all vertices of the Voronoi diagram (which are the circumcenters of the Delaunay facets) of the distance to the nearest site.
    # BUT, only if the circumcenter is inside the facet. If the circumcenter is outside, the Voronoi cell is unbounded (in the spherical sense, it touches the "infinity" of the hemisphere? No, on the sphere, if the circumcenter is outside, the Voronoi cell is still bounded, but the circumcenter is not the center of the largest inscribed circle).
    # Actually, if the circumcenter is outside the spherical triangle, the largest inscribed circle in that triangle has its center at the midpoint of the longest side?
    # Let's verify. In a Euclidean obtuse triangle, the circumcenter is outside. The point in the triangle farthest from the vertices is the midpoint of the longest side?
    # Distance from midpoint of longest side to the two endpoints is L/2. Distance to the third vertex is the altitude.
    # If the triangle is obtuse, the altitude to the longest side is shorter than L/2?
    # Yes. So the farthest point is the midpoint of the longest side.
    # So, if circumcenter is outside, the effective radius is the length of the longest side / 2.
    
    # Calculate longest side for each simplex
    d12 = _angular_distance(p1, p2)
    d23 = _angular_distance(p2, p3)
    d31 = _angular_distance(p3, p1)
    
    max_side = np.maximum(d12, np.maximum(d23, d31))
    effective_radii = np.where(inside, radii, max_side / 2.0)
    
    max_radius = np.max(effective_radii)
    
    # Find indices of facets contributing to the max radius (within a small tolerance)
    # We want the top few to update
    threshold = max_radius - 1e-6
    worst_indices = np.where(effective_radii >= threshold)[0]
    
    # If no indices (due to floating point), take the argmax
    if len(worst_indices) == 0:
        worst_indices = [np.argmax(effective_radii)]
    
    # Limit to top 5 to avoid too many simultaneous updates
    worst_indices = worst_indices[:5]
    
    facets_info = []
    for idx in worst_indices:
        s = simplices[idx]
        cc = c_unit[idx]
        is_in = inside[idx]
        facets_info.append((s[0], s[1], s[2], cc, is_in))
        
    return max_radius, facets_info

def construct():
    # Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    start_time = time.time()
    time_limit = 115.0
    
    step_size = 0.05
    prev_radius = _covering_radius_and_gradients(points)[0]
    
    while time.time() - start_time < time_limit:
        radius, facets_info = _covering_radius_and_gradients(points)
        
        if radius < prev_radius - 1e-9:
            prev_radius = radius
            step_size = min(step_size * 1.05, 0.1) # Increase slightly if improving
        else:
            step_size = max(step_size * 0.8, 0.001) # Decrease if stuck
            
        # If step size is too small, break
        if step_size < 0.001 and radius < 0.2: # ~11.5 degrees
             # Try a few more iterations with tiny steps
             pass
        if step_size < 1e-5:
            break

        # Aggregate gradients/updates
        # For each point, accumulate the direction to move
        updates = np.zeros_like(points)
        
        for i, j, k, cc, is_in in facets_info:
            p1, p2, p3 = points[i], points[j], points[k]
            
            if is_in:
                # Move away from circumcenter
                proj1 = cc - np.dot(cc, p1) * p1
                proj2 = cc - np.dot(cc, p2) * p2
                proj3 = cc - np.dot(cc, p3) * p3
                
                n1 = np.linalg.norm(proj1)
                n2 = np.linalg.norm(proj2)
                n3 = np.linalg.norm(proj3)
                
                if n1 > 1e-10: updates[i] += proj1 / n1
                if n2 > 1e-10: updates[j] += proj2 / n2
                if n3 > 1e-10: updates[k] += proj3 / n3
            else:
                # Move endpoints of longest side apart
                d12 = _angular_distance(p1, p2)
                d23 = _angular_distance(p2, p3)
                d31 = _angular_distance(p3, p1)
                
                if d12 >= d23 and d12 >= d31:
                    t1 = _normalize(p2 - np.dot(p1, p2)*p1)
                    t2 = _normalize(p1 - np.dot(p2, p1)*p2)
                    updates[i] -= t1
                    updates[j] -= t2
                elif d23 >= d12 and d23 >= d31:
                    t2 = _normalize(p3 - np.dot(p2, p3)*p2)
                    t3 = _normalize(p2 - np.dot(p3, p2)*p3)
                    updates[j] -= t2
                    updates[k] -= t3
                else:
                    t3 = _normalize(p1 - np.dot(p3, p1)*p3)
                    t1 = _normalize(p3 - np.dot(p1, p3)*p1)
                    updates[k] -= t3
                    updates[i] -= t1
        
        # Apply updates
        for idx in range(N):
            if np.linalg.norm(updates[idx]) > 0:
                u = _normalize(updates[idx])
                points[idx] = _normalize(points[idx] + step_size * u)
                
    return points
# EVOLVE-BLOCK-END
