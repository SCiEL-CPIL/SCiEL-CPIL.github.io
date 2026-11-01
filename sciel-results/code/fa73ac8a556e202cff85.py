import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def construct():
        start_time = time.time()
        time_limit = 115.0 # Leave 5s margin
        
        # Initial Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        best_score = np.inf
        best_pts = pts.copy()
        
        # Helper to compute covering radius
        def compute_covers(pts):
            try:
                hull = ConvexHull(pts)
                max_r = 0.0
                worst_facet = None
                worst_c = None
                for simplex in hull.simplices:
                    v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
                    # Circumcenter
                    # c is parallel to (v1-v2)x(v1-v3)
                    u = np.cross(v1 - v2, v1 - v3)
                    norm_u = np.linalg.norm(u)
                    if norm_u < 1e-10:
                        continue
                    c = u / norm_u
                    # Check sign: c should be such that the cap is the "small" one?
                    # Actually, the distance is arccos(c.v1).
                    # If the triangle is obtuse, the circumcenter might be on the "far" side.
                    # We want the radius of the cap that is empty.
                    # The empty cap is centered at c if c is the "inner" circumcenter.
                    # Let's just compute d = arccos(c.v1).
                    # If d > pi/2, the cap is larger than a hemisphere.
                    # We want the minimal cap containing the points? No, the covering radius is the max distance to nearest point.
                    # The "hole" is the region not covered.
                    # The center of the largest hole is the circumcenter.
                    # The radius is the distance to the vertices.
                    d = np.arccos(np.clip(c @ v1, -1.0, 1.0))
                    # If the triangle is "large", the circumcenter might be the "anti-center".
                    # We want the distance to be < pi/2 usually.
                    if d > np.pi/2:
                         d = np.pi - d # ? No.
                         # If c is the far circumcenter, the near one is -c.
                         # Distance to -c is arccos(-c.v1) = pi - arccos(c.v1).
                         # We want the smaller distance?
                         # The covering radius is the max distance from any point to nearest site.
                         # The farthest point in the Voronoi cell is the circumcenter.
                         # If the Voronoi cell is large, the circumcenter is far.
                         # We want to minimize this max distance.
                         # So we want the distance from the site to the circumcenter.
                         # If the triangle is obtuse, the circumcenter is outside the triangle.
                         # The distance is still arccos(c.v1).
                         # But we must ensure we pick the correct c.
                         # The two candidates are c and -c.
                         # The one that is the center of the empty cap is the one where the cap does not contain other points.
                         # For Delaunay, the circumcircle is empty.
                         # So we just take the distance.
                         # However, if the triangle is very large, the "empty cap" might be the one centered at -c?
                         # No, the Delaunay triangle corresponds to a Voronoi vertex.
                         # The Voronoi vertex is unique.
                         # It is the intersection of the bisectors.
                         # The distance is well-defined.
                    if d > max_r:
                        max_r = d
                        worst_facet = simplex
                        worst_c = c
                return max_r, worst_facet, worst_c
            except Exception:
                return np.inf, None, None

        # Main Loop
        step_size = 0.1
        for k in range(10000):
            if time.time() - start_time > time_limit:
                break
                
            max_r, facet, c = compute_covers(pts)
            if facet is None:
                break
            
            if max_r < best_score:
                best_score = max_r
                best_pts = pts.copy()
            
            # Move points of the worst facet towards c
            v1, v2, v3 = pts[facet[0]], pts[facet[1]], pts[facet[2]]
            
            # Spherical interpolation (slerp) or simple vector addition
            # Simple vector addition towards c
            for idx in facet:
                p = pts[idx]
                # Move towards c
                # Direction is (c - p) normalized? Or just c - p?
                # On sphere, gradient of distance to c is (c - p)/||c-p||?
                # Let's use a step along the geodesic.
                # Angle between p and c
                dot = np.clip(p @ c, -1.0, 1.0)
                theta = np.arccos(dot)
                if theta < 1e-6:
                    continue
                # Move by fraction of theta
                # new_p = normalize( (sin((1-alpha)*theta)/sin(theta)) * p + (sin(alpha*theta)/sin(theta)) * c )
                alpha = step_size
                # Simplified: p_new = normalize(p + alpha * (c - p))
                # This is an approximation for small steps.
                p_new = p + step_size * (c - p)
                pts[idx] = p_new / np.linalg.norm(p_new)
            
            # Decay step size?
            # step_size *= 0.99
            # Or keep constant and rely on convergence?
            # Let's try to decrease step size slowly.
            step_size = max(0.01, step_size * 0.995)
            
        return best_pts
