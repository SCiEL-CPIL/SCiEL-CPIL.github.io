import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def _spherical_circumradius(v1, v2, v3):
    """
    Compute the spherical circumradius of the triangle formed by v1, v2, v3 on the unit sphere.
    Returns the angle in radians.
    """
    d1 = v1 - v2
    d2 = v2 - v3
    c = np.cross(d1, d2)
    norm_c = np.linalg.norm(c)
    if norm_c < 1e-10:
        return np.pi / 2
    c = c / norm_c
    
    dot = np.dot(c, v1)
    if dot < 0:
        c = -c
        dot = -dot
    
    dot = np.clip(dot, -1.0, 1.0)
    return np.arccos(dot)

def _covering_radius(points):
    """
    Compute the covering radius (max spherical circumradius of hull facets) in degrees.
    """
    try:
        hull = ConvexHull(points)
        max_r = 0.0
        for simplex in hull.simplices:
            v1 = points[simplex[0]]
            v2 = points[simplex[1]]
            v3 = points[simplex[2]]
            r = _spherical_circumradius(v1, v2, v3)
            if r > max_r:
                max_r = r
        return np.degrees(max_r)
    except Exception:
        return 180.0

def _fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_points(n, seed=42):
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(n, 3))
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    return points / norms

def _spherical_centroid(points):
    """
    Compute the spherical centroid of a set of points.
    """
    if len(points) == 0:
        return None
    c = np.sum(points, axis=0)
    norm_c = np.linalg.norm(c)
    if norm_c < 1e-10:
        # Points are spread out, centroid is undefined/zero. 
        # Return a random point or the first point? 
        # In Lloyd's, if a cell is empty or centroid is zero, we can keep the point or move it randomly.
        # Here we return None to signal this.
        return None
    return c / norm_c

def _lloyd_iteration(points, max_iter=100, tol=1e-8):
    """
    Perform Spherical Lloyd's algorithm iterations.
    """
    pts = points.copy()
    for _ in range(max_iter):
        # Compute Voronoi cells
        try:
            # We need to assign each point on the sphere to the nearest point in pts.
            # This is equivalent to finding the Voronoi diagram.
            # We can approximate by checking all pairs, but that is O(N^2 * GridSize).
            # A better way is to use the convex hull of the points and their antipodes?
            # Actually, for N=50, we can just compute the nearest neighbor for a set of test points?
            # No, Lloyd's requires the centroid of the Voronoi cell.
            # The Voronoi cell of a point on the sphere is the set of points on the sphere closer to it than to any other point.
            # The spherical centroid of this cell can be computed as the normalized sum of the vertices of the Voronoi cell?
            # No, it's the normalized sum of the points in the cell (integral).
            
            # Efficient way:
            # 1. Compute the convex hull of the points.
            # 2. The Voronoi cell of a point corresponds to the face of the dual polytope.
            # This is complex.
            
            # Alternative: 
            # Use the fact that the spherical Voronoi cell of p_i is the intersection of hemispheres centered at p_i with radius half the distance to p_j.
            # We can approximate the centroid by taking the normalized sum of the points p_i weighted by the solid angle of the cell?
            # Or simply: 
            # For each point p_i, find the set of points q on the sphere such that q is closer to p_i than to any other p_j.
            # This is hard to do analytically for the centroid.
            
            # Let's use a simpler heuristic often used in these problems:
            # Move each point towards the centroid of its nearest neighbors? No, that's not Lloyd's.
            
            # Let's stick to a simpler local optimization: 
            # Gradient descent on the covering radius?
            # Or just use Nelder-Mead on the exact covering radius, which is what the current program does.
            # The current program's weakness is likely getting stuck in local minima.
            
            # Let's implement a "Kick" strategy with Nelder-Mead.
            # 1. Optimize with NM.
            # 2. Pick a random point and move it significantly.
            # 3. Re-optimize.
            
            pass
        except Exception:
            pass
        
        # Since implementing full spherical Lloyd's is complex and error-prone without a library,
        # we will rely on Multi-Start Nelder-Mead with aggressive restarts.
        break
        
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0 # 5s safety margin
    
    best_points = None
    best_score = float('inf')
    
    # Initializations
    inits = [
        _fibonacci_points(N),
        _random_points(N, seed=1),
        _random_points(N, seed=2),
        _random_points(N, seed=3),
        _random_points(N, seed=4),
        _random_points(N, seed=5),
    ]
    
    def objective(x):
        pts = x.reshape(N, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        if np.any(norms < 1e-6):
            return 180.0
        pts = pts / norms
        return _covering_radius(pts)
    
    for init in inits:
        if time.time() > deadline:
            break
            
        x0 = init.flatten()
        
        try:
            # First pass: coarse optimization
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
            
            if res.fun < best_score:
                best_points = res.x.reshape(N, 3)
                best_score = res.fun
                
        except Exception:
            pass
            
        # If we have time, do a refinement on this specific solution
        if time.time() < deadline - 10.0:
            try:
                # Second pass: fine optimization
                res2 = minimize(objective, res.x, method='Nelder-Mead', 
                               options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8, 'initial_simplex': 1e-3})
                if res2.fun < best_score:
                    best_points = res2.x.reshape(N, 3)
                    best_score = res2.fun
            except Exception:
                pass

    # Final polish on the global best
    if best_points is not None and time.time() < deadline - 5.0:
        x0 = best_points.flatten()
        try:
            res_final = minimize(objective, x0, method='Nelder-Mead', 
                               options={'maxiter': 50, 'xatol': 1e-10, 'fatol': 1e-10, 'initial_simplex': 1e-4})
            if res_final.fun < best_score:
                best_points = res_final.x.reshape(N, 3)
                best_score = res_final.fun
        except Exception:
            pass

    # Final normalization
    if best_points is None:
        return _fibonacci_points(N)
        
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points
