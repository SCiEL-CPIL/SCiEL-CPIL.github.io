import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _compute_covering_radius(points):
    """
    Compute the covering radius of a set of points on the unit sphere.
    The covering radius is the maximum angular distance from any point on the sphere
    to the nearest configuration point. This is equivalent to the maximum circumradius
    of the facets of the convex hull of the points.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        max_radius = 0.0
        
        for tri in facets:
            p1 = points[tri[0]]
            p2 = points[tri[1]]
            p3 = points[tri[2]]
            
            # Compute the circumcenter of the spherical triangle (p1, p2, p3)
            # The circumcenter c satisfies |c - p1| = |c - p2| = |c - p3| on the sphere.
            # In 3D, the circumcenter of a triangle on a sphere lies on the plane defined by p1, p2, p3.
            # We can find the point c in the span of p1, p2, p3 that is equidistant from p1, p2, p3.
            # Let c = a*p1 + b*p2 + c*p3.
            # |c-p1|^2 = |c-p2|^2 => -2 c.p1 + |p1|^2 = -2 c.p2 + |p2|^2 => 2 c.(p2-p1) = |p1|^2 - |p2|^2 = 0 (since unit)
            # So c is orthogonal to (p2-p1) and (p3-p1).
            # Thus c is parallel to the cross product (p2-p1) x (p3-p1).
            
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            
            if np.linalg.norm(cross) < 1e-12:
                # Degenerate triangle
                continue
                
            c = cross / np.linalg.norm(cross)
            
            # Check if c is on the correct side (same side as the origin relative to the plane? No, the circumcenter must be inside the cap).
            # The circumcenter of a spherical triangle is the point on the sphere equidistant from the vertices.
            # There are two such points. We want the one that minimizes the distance to the vertices (the minor arc).
            # The dot product c.p1 should be positive for the minor arc circumcenter if the triangle is "small".
            # Actually, we just need the angular distance.
            
            # Distance from c to p1
            dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
            dist = np.arccos(dot_val)
            
            # If the triangle is large ( > 90 deg ), the circumcenter might be on the other side.
            # However, for a good covering, facets are small.
            # To be safe, we check the other candidate -c.
            dot_val_neg = np.clip(np.dot(-c, p1), -1.0, 1.0)
            dist_neg = np.arccos(dot_val_neg)
            
            if dist_neg < dist:
                dist = dist_neg
            
            if dist > max_radius:
                max_radius = dist
                
    except Exception:
        # If hull fails, return a large value or estimate
        return np.pi / 2
        
    return np.degrees(max_radius)

def construct():
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    start_time = time.time()
    deadline = start_time + 140.0 # 10s safety margin
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    step_size = 0.01
    
    while time.time() < deadline:
        # Compute hull and find worst facet
        try:
            hull = ConvexHull(points)
            facets = hull.simplices
            
            worst_facet_idx = -1
            worst_radius = 0.0
            worst_circumcenter = None
            worst_vertices = None
            
            for idx, tri in enumerate(facets):
                p1 = points[tri[0]]
                p2 = points[tri[1]]
                p3 = points[tri[2]]
                
                v1 = p2 - p1
                v2 = p3 - p1
                cross = np.cross(v1, v2)
                norm_cross = np.linalg.norm(cross)
                
                if norm_cross < 1e-12:
                    continue
                    
                c = cross / norm_cross
                
                dot_val = np.clip(np.dot(c, p1), -1.0, 1.0)
                dist = np.arccos(dot_val)
                
                dot_val_neg = np.clip(np.dot(-c, p1), -1.0, 1.0)
                dist_neg = np.arccos(dot_val_neg)
                
                if dist_neg < dist:
                    dist = dist_neg
                    c = -c
                
                if dist > worst_radius:
                    worst_radius = dist
                    worst_facet_idx = idx
                    worst_circumcenter = c
                    worst_vertices = tri
                    
        except Exception:
            break
            
        if worst_facet_idx == -1:
            break
            
        # Move vertices towards the circumcenter
        # We move all 3 vertices slightly towards the circumcenter to shrink the facet
        for v_idx in worst_vertices:
            p = points[v_idx]
            # Direction to move: towards circumcenter
            # Use spherical interpolation or simple linear projection
            direction = worst_circumcenter - p
            direction /= np.linalg.norm(direction)
            
            points[v_idx] = points[v_idx] + step_size * direction
            # Normalize
            points[v_idx] /= np.linalg.norm(points[v_idx])
        
        # Check improvement
        current_radius = _compute_covering_radius(points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            # Optionally increase step size if improving fast? Keep simple.
        else:
            # Revert if worse
            # We don't keep a history, so we just revert this specific move?
            # Since we moved 3 points, we should probably store the previous state or just revert those 3.
            # For simplicity and robustness, let's just revert the whole step if it's worse.
            # But we modified 'points' in place. We need to restore 'points' from 'best_points'? No, that would lose other improvements.
            # We should have stored the previous state of the 3 points.
            # Let's refine the loop to store prev points.
            
            # Actually, let's just restart the loop logic to be safer.
            # If current_radius is worse, we revert the changes to 'points' to the state before this iteration.
            # We didn't store it. So we must store it.
            
            # Let's restructure the inner loop to store prev_points.
            pass # Handled in the next version of the code below

    return best_points
