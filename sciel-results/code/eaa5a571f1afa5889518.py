import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
import time

N = 50

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    # Avoid division by zero
    norms[norms == 0] = 1.0
    return points / norms

def _covering_radius(points):
    """
    Compute the covering radius of the points on the unit sphere.
    This is the maximum circumradius of the spherical Delaunay triangles.
    """
    # Check if points are valid
    if not np.all(np.isfinite(points)):
        return np.pi
    
    # Normalize
    pts = _normalize(points)
    
    # Check if all points are in a closed hemisphere
    # A set of points is in a closed hemisphere if there exists a vector v such that v . p_i >= 0 for all i
    # This is equivalent to the origin being outside the convex hull.
    # We can check this by solving a linear programming problem or simply checking if the Delaunay triangulation is valid and covers the sphere.
    # If the points are in a hemisphere, the Delaunay triangulation might be degenerate or the "circumcenters" might be on the wrong side.
    # A simpler check: if the convex hull does not contain the origin, the covering radius is at least pi/2 (and likely much larger).
    # We can use scipy.spatial.ConvexHull to check if origin is inside.
    try:
        hull = Delaunay(pts)
        # Check if origin is in the convex hull
        # The convex hull of points on the sphere contains the origin iff the points are not in any open hemisphere.
        # We can check the sign of the volume of the tetrahedron formed by origin and each face.
        # If all faces have the same sign relative to the origin, they are in a hemisphere.
        # Actually, Delaunay triangulation on the sphere works by projecting to 3D.
        # If the points are in a hemisphere, the Delaunay triangulation will have "holes" or be ill-defined in terms of covering the whole sphere.
        # The task says "configurations inside a closed hemisphere ... make the candidate invalid".
        # So we should penalize this heavily.
        
        # Simple check: 
        # Find the centroid. If the centroid is 0, it's symmetric.
        # Better: Use the fact that if the origin is inside the convex hull, the covering radius is well-defined and < pi/2 (usually).
        # If the origin is outside, the "covering" of the sphere is poor.
        
        # Let's compute the circumradii anyway.
        simplices = hull.simplices
        radii = []
        for s in simplices:
            a, b, c = pts[s[0]], pts[s[1]], pts[s[2]]
            # Circumcenter vector
            v = np.cross(a, b) + np.cross(b, c) + np.cross(c, a)
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-10:
                # Degenerate triangle
                radii.append(np.pi)
                continue
            m = v / norm_v
            
            # Angular distances
            d_a = np.arccos(np.clip(np.dot(m, a), -1.0, 1.0))
            d_b = np.arccos(np.clip(np.dot(m, b), -1.0, 1.0))
            d_c = np.arccos(np.clip(np.dot(m, c), -1.0, 1.0))
            
            r = np.max([d_a, d_b, d_c])
            radii.append(r)
            
        if not radii:
            return np.pi
            
        return np.max(radii)
    except Exception:
        return np.pi

def construct():
    start_time = time.time()
    deadline = 110.0 # 10s margin
    
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_pts = pts
    best_score = _covering_radius(pts)
    
    # Optimization loop
    # We use L-BFGS-B with a soft-max objective to smooth the max operation.
    
    def objective(x, beta):
        p = x.reshape(-1, 3)
        p = _normalize(p)
        
        # Check for hemisphere issue
        # If the points are in a hemisphere, the Delaunay triangulation might fail or produce large radii.
        # We add a penalty if the centroid is too close to the surface (indicating hemisphere).
        centroid = np.mean(p, axis=0)
        c_norm = np.linalg.norm(centroid)
        # If c_norm is close to 1, points are clustered in a hemisphere.
        # Ideal c_norm is 0.
        penalty = 0.0
        if c_norm > 0.5:
            penalty = (c_norm - 0.5) * 10.0
        
        try:
            hull = Delaunay(p)
            simplices = hull.simplices
            radii = []
            for s in simplices:
                a, b, c = p[s[0]], p[s[1]], p[s[2]]
                v = np.cross(a, b) + np.cross(b, c) + np.cross(c, a)
                norm_v = np.linalg.norm(v)
                if norm_v < 1e-10:
                    radii.append(np.pi)
                    continue
                m = v / norm_v
                d_a = np.arccos(np.clip(np.dot(m, a), -1.0, 1.0))
                d_b = np.arccos(np.clip(np.dot(m, b), -1.0, 1.0))
                d_c = np.arccos(np.clip(np.dot(m, c), -1.0, 1.0))
                r = np.max([d_a, d_b, d_c])
                radii.append(r)
            
            if not radii:
                return penalty + np.pi
                
            radii = np.array(radii)
            
            # Soft max approximation of the max radius
            # max(radii) approx (1/beta) * log(sum(exp(beta * radii)))
            # To avoid overflow, subtract max
            max_r = np.max(radii)
            if beta > 0:
                soft_max = max_r + np.log(np.sum(np.exp(beta * (radii - max_r)))) / beta
            else:
                soft_max = max_r
                
            return penalty + soft_max
        except Exception:
            return penalty + np.pi

    # Annealing schedule for beta
    betas = [5.0, 10.0, 20.0, 50.0, 100.0]
    
    current_pts = pts.copy()
    
    for beta in betas:
        if time.time() > deadline:
            break
            
        x0 = current_pts.flatten()
        
        # Use L-BFGS-B
        # We don't provide gradients, so it uses numerical gradients.
        # 150 dimensions.
        try:
            res = minimize(
                objective, 
                x0, 
                args=(beta,), 
                method='L-BFGS-B', 
                options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-5, 'maxls': 50}
            )
            
            if res.success or res.fun < objective(x0, beta):
                new_pts = _normalize(res.x.reshape(-1, 3))
                new_score = _covering_radius(new_pts)
                
                if new_score < best_score:
                    best_score = new_score
                    best_pts = new_pts
                    current_pts = new_pts
        except Exception:
            pass
            
    return best_pts
