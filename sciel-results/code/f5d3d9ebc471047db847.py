import numpy as np
import scipy.spatial
import time

N = 130

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # 1. Initial Configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = np.inf
    
    def calc_covering_radius(pts):
        # Compute convex hull
        try:
            hull = scipy.spatial.ConvexHull(pts)
        except Exception:
            return np.inf
        
        max_rad = 0.0
        facets = hull.simplices
        for f in facets:
            v1, v2, v3 = pts[f[0]], pts[f[1]], pts[f[2]]
            # Circumcenter of spherical triangle
            # c is proportional to (v1 - v3) x (v2 - v3)
            c_vec = np.cross(v1 - v3, v2 - v3)
            c_norm = np.linalg.norm(c_vec)
            if c_norm < 1e-10:
                continue
            c = c_vec / c_norm
            # Distance from c to v1
            dot = np.dot(c, v1)
            # Clamp for numerical stability
            dot = np.clip(dot, -1.0, 1.0)
            rad = np.arccos(dot)
            if rad > max_rad:
                max_rad = rad
        return max_rad

    best_radius = calc_covering_radius(points)
    
    # 2. Local Search
    step_size = 0.05
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline:
        # Find worst facet
        try:
            hull = scipy.spatial.ConvexHull(points)
            facets = hull.simplices
        except Exception:
            break
        
        max_rad = 0.0
        worst_f = None
        worst_c = None
        for f in facets:
            v1, v2, v3 = points[f[0]], points[f[1]], points[f[2]]
            c_vec = np.cross(v1 - v3, v2 - v3)
            c_norm = np.linalg.norm(c_vec)
            if c_norm < 1e-10:
                continue
            c = c_vec / c_norm
            dot = np.clip(np.dot(c, v1), -1.0, 1.0)
            rad = np.arccos(dot)
            if rad > max_rad:
                max_rad = rad
                worst_f = f
                worst_c = c
        
        if worst_f is None:
            break
            
        # Move vertices of worst facet towards circumcenter
        new_points = points.copy()
        v1, v2, v3 = new_points[worst_f[0]], new_points[worst_f[1]], new_points[worst_f[2]]
        
        # Move towards c
        dir1 = worst_c - v1
        dir2 = worst_c - v2
        dir3 = worst_c - v3
        
        new_points[worst_f[0]] = v1 + step_size * dir1
        new_points[worst_f[1]] = v2 + step_size * dir2
        new_points[worst_f[2]] = v3 + step_size * dir3
        
        # Normalize
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        new_radius = calc_covering_radius(new_points)
        
        if new_radius < best_radius - 1e-12:
            best_radius = new_radius
            best_points = new_points.copy()
            points = new_points.copy()
            no_improve_count = 0
            # Optionally reduce step size slightly if improving consistently?
            # Or keep it constant.
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Random perturbation to escape local minimum
                points += np.random.normal(0, 0.01, points.shape)
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                no_improve_count = 0
                # Re-evaluate
                best_radius = calc_covering_radius(points)
                best_points = points.copy()

    return best_points
