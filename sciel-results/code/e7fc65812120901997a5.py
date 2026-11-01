import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds to be safe
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = np.inf
    
    def compute_covering_radius(pts):
        # Compute convex hull
        try:
            hull = ConvexHull(pts)
        except Exception:
            return np.inf
        
        facets = hull.simplices
        max_r = 0.0
        
        for f in facets:
            a, b, c = pts[f[0]], pts[f[1]], pts[f[2]]
            # Compute circumcenter of triangle a, b, c on the sphere
            # The circumcenter o satisfies o.a = o.b = o.c = cos(r)
            # o is in the plane of a, b, c.
            # o = alpha*a + beta*b + gamma*c
            # We can solve for o using the fact that o is equidistant from a, b, c.
            # (o-a)^2 = (o-b)^2 => -2o.a + a.a = -2o.b + b.b => o.(b-a) = 0.5(b.b - a.a) = 0
            # So o is in the plane spanned by a and b, perpendicular to b-a? No.
            # o is in the plane of the triangle. The plane normal is n = (b-a) x (c-a).
            # o = n x (n x a) / |n x (n x a)|^2 ? No.
            
            # Better: o is the intersection of the sphere and the plane of the triangle.
            # The plane equation is n . x = d, where n = (b-a) x (c-a), d = n . a.
            # We want x on the sphere (|x|=1) and n . x = d.
            # Let x = k * n + p, where p is the projection of the origin onto the plane?
            # Actually, the circumcenter is the point on the sphere closest to the plane's intersection with the sphere's center? No.
            
            # Standard method:
            # The circumcenter o is the solution to:
            # o . a = o . b = o . c = cos(r)
            # |o| = 1
            # This is a linear system for o if we fix cos(r), but r is unknown.
            
            # Alternative:
            # The circumcenter is the normalized vector:
            # o = (a * |a|^2 * (b x c) + b * |b|^2 * (c x a) + c * |c|^2 * (a x b)) / |...|
            # Since |a|=|b|=|c|=1:
            # o = (a * (b x c) + b * (c x a) + c * (a x b)) / |...|
            
            v = a * np.cross(b, c) + b * np.cross(c, a) + c * np.cross(a, b)
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-10:
                continue # Degenerate
            o = v / norm_v
            
            # Ensure o is on the "correct" side of the triangle (same side as the origin? No, opposite)
            # The circumcenter should be such that the triangle is "seen" from the center.
            # Check dot product with normal
            n = np.cross(b - a, c - a)
            if np.dot(n, o) < 0:
                o = -o
                
            r_facet = np.arccos(np.clip(np.dot(o, a), -1.0, 1.0))
            if r_facet > max_r:
                max_r = r_facet
                
        return max_r

    # Main optimization loop
    step_size = 0.01
    while time.time() < deadline:
        radius = compute_covering_radius(points)
        if radius < best_radius:
            best_radius = radius
            best_points = points.copy()
            
        # Compute gradients for the worst facet
        # We need to find the worst facet again
        try:
            hull = ConvexHull(points)
        except Exception:
            break
            
        facets = hull.simplices
        worst_facet = None
        worst_r = 0.0
        worst_circumcenter = None
        
        for f in facets:
            a, b, c = points[f[0]], points[f[1]], points[f[2]]
            v = a * np.cross(b, c) + b * np.cross(c, a) + c * np.cross(a, b)
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-10:
                continue
            o = v / norm_v
            n = np.cross(b - a, c - a)
            if np.dot(n, o) < 0:
                o = -o
            r_facet = np.arccos(np.clip(np.dot(o, a), -1.0, 1.0))
            if r_facet > worst_r:
                worst_r = r_facet
                worst_facet = f
                worst_circumcenter = o
                
        if worst_facet is None:
            break
            
        # Update the 3 points of the worst facet
        # Gradient of r with respect to a is -o / sin(r)
        # sin(r) = sqrt(1 - cos(r)^2) = sqrt(1 - (o.a)^2)
        idxs = worst_facet
        for idx in idxs:
            p = points[idx]
            cos_r = np.dot(worst_circumcenter, p)
            sin_r = np.sqrt(1.0 - cos_r * cos_r)
            if sin_r < 1e-10:
                continue
            grad = -worst_circumcenter / sin_r
            # Project to tangent space
            grad_tangent = grad - np.dot(grad, p) * p
            # Update
            points[idx] = p - step_size * grad_tangent
            # Re-normalize
            points[idx] /= np.linalg.norm(points[idx])
            
        # Occasionally, do a small random perturbation to escape local minima
        if np.random.random() < 0.05:
            idx = np.random.randint(N)
            p = points[idx]
            # Random direction on tangent space
            d = np.random.randn(3)
            d -= np.dot(d, p) * p
            d /= np.linalg.norm(d)
            points[idx] = p + 0.01 * d
            points[idx] /= np.linalg.norm(points[idx])

    return best_points
