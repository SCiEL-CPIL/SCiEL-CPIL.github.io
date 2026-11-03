import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def _normalize(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)
    return v / n

def _time_ok(start, limit=145.0):
    return (time.time() - start) < limit

def _covering_radius(pts):
    # Compute covering radius using convex hull of points on sphere
    # The covering radius is the max circumradius of the spherical Voronoi cells
    # Which corresponds to the max circumradius of the facets of the convex hull
    try:
        hull = ConvexHull(pts)
        max_r = 0.0
        for simplex in hull.simplices:
            # Get the points forming this facet
            facet_pts = pts[simplex]
            # Find the circumcenter of this facet (on the sphere)
            # For a spherical triangle, the circumcenter is the normalized sum of the vertices
            # weighted by the opposite edge lengths? No, simpler:
            # The circumcenter of a spherical triangle is the normalized sum of the vertices
            # IF the triangle is not too large. But for robustness, we can use the fact that
            # the circumcenter is the intersection of the perpendicular bisectors.
            # However, a simpler and robust method for the circumradius of a spherical triangle
            # defined by unit vectors a, b, c is:
            # cos(r) = (a dot b + b dot c + c dot a - 1) / (2 * (a dot b * b dot c * c dot a - a dot b - b dot c - c dot a + 1))
            # This is complex. Let's use a different approach.
            
            # Alternative: The circumradius is the max angular distance from the circumcenter to the vertices.
            # We can find the circumcenter by solving for the point x on the sphere that is equidistant from a, b, c.
            # This is equivalent to finding x such that x dot a = x dot b = x dot c.
            # This means x is orthogonal to (a-b) and (b-c).
            # So x is parallel to (a-b) x (b-c).
            
            a, b, c = facet_pts
            v1 = a - b
            v2 = b - c
            x = np.cross(v1, v2)
            norm_x = np.linalg.norm(x)
            if norm_x < 1e-10:
                # Degenerate, use centroid
                x = a + b + c
                norm_x = np.linalg.norm(x)
            if norm_x < 1e-10:
                continue
            x = x / norm_x
            
            # Ensure x is on the same side of the facet as the origin? 
            # The facet is on the convex hull of the points. The origin is inside the hull.
            # The circumcenter should be on the sphere.
            # The angular distance from x to a is arccos(x dot a).
            # We need to check if x is the "correct" circumcenter (inside the cell).
            # For a convex hull of points on the sphere, the facet normal points outward.
            # The circumcenter of the spherical cap is the normalized facet normal?
            # No, the facet normal is the cross product of edges, which is parallel to x.
            # Let's check the sign. The origin is at 0. The facet is at distance d > 0 from origin.
            # The normal n = (a-b)x(b-c) points outward if the points are ordered correctly.
            # Actually, the convex hull normal points away from the origin.
            # So x (normalized) is the point on the sphere closest to the facet plane?
            # No, the point on the sphere farthest from the facet plane is the antipode of the normal?
            
            # Let's just compute the max distance from x to the vertices.
            # And also from -x to the vertices, and take the min of the two max distances?
            # No, the circumcenter of the spherical triangle is unique (if < pi/2).
            # The point x derived from the cross product is the circumcenter.
            
            d1 = np.arccos(np.clip(np.dot(x, a), -1, 1))
            d2 = np.arccos(np.clip(np.dot(x, b), -1, 1))
            d3 = np.arccos(np.clip(np.dot(x, c), -1, 1))
            r = max(d1, d2, d3)
            
            # Check if the circumcenter is inside the triangle.
            # If the triangle is obtuse, the circumcenter might be outside.
            # But for covering radius, we care about the empty cap.
            # The circumradius of the facet is the radius of the smallest cap containing the facet.
            # If the circumcenter is outside, the "cap" is not empty?
            # Actually, the Voronoi cell is the set of points closer to a than any other.
            # The circumradius is the distance to the farthest vertex of the Voronoi cell.
            # The vertices of the Voronoi cell are the circumcenters of the facets.
            # So we just need the max circumradius of all facets.
            
            if r > max_r:
                max_r = r
        return max_r
    except Exception:
        # Fallback: max distance to nearest neighbor
        # This is an underestimate of covering radius
        return 0.0

def _covering_radius_fast(pts):
    # Faster approximation using a grid
    # This is less accurate but much faster
    # We'll use a grid of points on the sphere and compute the max distance to nearest point
    # But this is also slow if the grid is fine.
    # Let's stick with the hull method but optimize it.
    pass

def _fibonacci_spiral(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _local_refine(pts, start_time, time_limit=145.0):
    pts = _normalize(pts)
    best_pts = pts.copy()
    best_r = _covering_radius(pts)
    
    step_size = 0.1
    for it in range(1000):
        if not _time_ok(start_time, time_limit):
            break
        
        # 1. Find the worst facet
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
            
        max_r = 0
        worst_simplex = None
        worst_circumcenter = None
        
        for simplex in hull.simplices:
            facet_pts = pts[simplex]
            a, b, c = facet_pts
            v1 = a - b
            v2 = b - c
            x = np.cross(v1, v2)
            norm_x = np.linalg.norm(x)
            if norm_x < 1e-10:
                continue
            x = x / norm_x
            
            d1 = np.arccos(np.clip(np.dot(x, a), -1, 1))
            d2 = np.arccos(np.clip(np.dot(x, b), -1, 1))
            d3 = np.arccos(np.clip(np.dot(x, c), -1, 1))
            r = max(d1, d2, d3)
            
            if r > max_r:
                max_r = r
                worst_simplex = simplex
                worst_circumcenter = x
        
        if worst_simplex is None:
            break
            
        # 2. Move the points in the worst simplex towards the circumcenter
        # This reduces the circumradius of that facet
        for idx in worst_simplex:
            p = pts[idx]
            # Move p towards worst_circumcenter
            # Step size should be proportional to the distance
            d = np.arccos(np.clip(np.dot(p, worst_circumcenter), -1, 1))
            if d < 1e-6:
                continue
            # Spherical interpolation
            sin_d = np.sin(d)
            if sin_d < 1e-6:
                continue
            new_p = (np.sin((1-step_size)*d) * p + np.sin(step_size*d) * worst_circumcenter) / sin_d
            pts[idx] = new_p
        
        pts = _normalize(pts)
        
        # 3. Pairwise repulsion to prevent clustering
        # Simple pass: for each pair, if too close, push apart
        for i in range(N):
            for j in range(i+1, N):
                d = np.arccos(np.clip(np.dot(pts[i], pts[j]), -1, 1))
                if d < 0.1: # Too close
                    # Push apart
                    v = pts[i] - pts[j]
                    norm_v = np.linalg.norm(v)
                    if norm_v > 1e-6:
                        v = v / norm_v
                        pts[i] += 0.01 * v
                        pts[j] -= 0.01 * v
        
        pts = _normalize(pts)
        
        # 4. Check improvement
        current_r = _covering_radius(pts)
        if current_r < best_r:
            best_r = current_r
            best_pts = pts.copy()
        else:
            # No improvement, revert and reduce step size?
            # Or just break?
            # Let's try to break if no improvement for a while
            # For now, just continue but maybe reduce step size
            step_size *= 0.95
            if step_size < 0.001:
                break
        
        # Early stop if we reach a good value
        if best_r < 0.197: # ~11.3 degrees
            break
            
    return best_pts, best_r

def construct():
    start_time = time.time()
    time_limit = 145.0
    
    best_pts = None
    best_r = np.inf
    
    # Multi-restart
    for restart in range(20):
        if not _time_ok(start_time, time_limit):
            break
            
        # Different initializations
        if restart == 0:
            pts = _fibonacci_spiral(N)
        else:
            # Random with some structure
            pts = np.random.randn(N, 3)
            pts = _normalize(pts)
            # Add some noise to fibonacci
            if restart % 2 == 0:
                base = _fibonacci_spiral(N)
                noise = np.random.randn(N, 3) * 0.1
                pts = _normalize(base + noise)
        
        refined_pts, r = _local_refine(pts, start_time, time_limit)
        
        if r < best_r:
            best_r = r
            best_pts = refined_pts
            
        # If we have a very good result, stop
        if best_r < 0.197:
            break
            
    if best_pts is None:
        # Fallback
        best_pts = _fibonacci_spiral(N)
        
    return best_pts
