import numpy as np
import time
import scipy.spatial
import scipy.optimize

N = 50
TIME_LIMIT = 120.0  # Safety margin from 150s

def _compute_circumradii(points):
    """
    Compute the squared circumradii of the facets of the convex hull of points.
    Returns the sum of squared radii and the max squared radius.
    """
    try:
        hull = scipy.spatial.ConvexHull(points)
    except Exception:
        # If hull fails, return a large penalty
        return 1e10, 1e10
    
    sum_sq_rad = 0.0
    max_sq_rad = 0.0
    
    # For each facet (simplex)
    for simplex in hull.simplices:
        v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        # Compute the normal to the plane containing v1, v2, v3
        # n = (v2 - v1) x (v3 - v1)
        n = np.cross(v2 - v1, v3 - v1)
        norm_n = np.linalg.norm(n)
        
        if norm_n < 1e-12:
            # Degenerate facet
            sum_sq_rad += 1e10
            max_sq_rad = max(max_sq_rad, 1e10)
            continue
            
        c = n / norm_n
        
        # The circumcenter on the sphere is c or -c.
        # We want the one closest to the vertices?
        # Actually, the center of the circle is the pole.
        # The angular distance from c to v1 is the radius.
        # dot = c . v1
        dot = np.dot(c, v1)
        dot = np.clip(dot, -1.0, 1.0)
        
        # If dot is negative, the center is on the other side?
        # The circumradius is arccos(abs(dot))? No.
        # The vertices are on the unit sphere. The plane cuts the sphere.
        # The distance from the origin to the plane is |c . v1|? No, c is unit normal.
        # The distance from origin to plane is d. c is normal.
        # The radius of the circle of intersection is sqrt(1 - d^2).
        # The angular radius (distance on sphere from center to vertex) is arccos(d)?
        # Let's check.
        # If the plane is z = d, normal is (0,0,1).
        # Vertices have z = d.
        # Distance from (0,0,1) to (x,y,d) is arccos(d).
        # So yes, angular radius is arccos(|c . v1|) if we consider the cap.
        # Wait, the "center" of the cap is the pole.
        # The angular distance is arccos(c . v1).
        # If c . v1 < 0, the pole is on the opposite side, distance is arccos(c . v1) > pi/2.
        # But the cap is usually defined by the smaller angle.
        # However, for covering, we care about the distance to the nearest point.
        # The "worst" point in the facet is the circumcenter.
        # The distance from the circumcenter to the vertices is the circumradius.
        # So we want to minimize this radius.
        # The radius is arccos(c . v1) if c . v1 > 0, else arccos(-c . v1)?
        # Actually, the circumcenter is the point on the sphere farthest from the plane?
        # No, the circumcenter of the spherical triangle is the pole of the plane.
        # There are two poles. The one "inside" the triangle is the relevant one.
        # For a small triangle, the pole is close to the vertices.
        # So c . v1 should be positive if c is chosen correctly.
        # We can ensure c points towards the vertices by checking sign.
        if dot < 0:
            c = -c
            dot = -dot
            
        # Now dot is in [0, 1]
        # Angular radius theta = arccos(dot)
        # theta^2 = arccos(dot)^2
        theta = np.arccos(dot)
        theta_sq = theta * theta
        
        sum_sq_rad += theta_sq
        if theta_sq > max_sq_rad:
            max_sq_rad = theta_sq
            
    return sum_sq_rad, max_sq_rad

def _objective(x):
    """
    Objective function for scipy.optimize.
    x is a flat array of shape (3*N,).
    """
    points = x.reshape(N, 3)
    
    # Penalty for not being on the unit sphere
    norms = np.linalg.norm(points, axis=1)
    sphere_penalty = np.sum((norms - 1.0)**2) * 100.0
    
    # Compute covering radius proxy
    sum_sq_rad, _ = _compute_circumradii(points)
    
    return sum_sq_rad + sphere_penalty

def _optimize_from_start(start_points):
    """
    Run local optimization from a given start.
    """
    x0 = start_points.flatten()
    
    # Use BFGS
    # Bounded? No, but penalty keeps them close.
    # We can use L-BFGS-B if we want to bound coordinates, but penalty is fine.
    
    start_time = time.time()
    try:
        res = scipy.optimize.minimize(
            _objective, 
            x0, 
            method='BFGS', 
            options={'maxiter': 100, 'disp': False}
        )
    except Exception:
        return start_points, _objective(x0)
    
    elapsed = time.time() - start_time
    x_opt = res.x.reshape(N, 3)
    
    # Normalize to unit sphere
    x_opt = x_opt / np.linalg.norm(x_opt, axis=1, keepdims=True)
    
    score = _objective(x_opt.flatten())
    
    return x_opt, score

def construct():
    start_time = time.time()
    best_points = None
    best_score = np.inf
    
    # Generate multiple starting points
    starts = []
    
    # 1. Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    starts.append(fib_points)
    
    # 2. Random points
    for _ in range(5):
        rand_points = np.random.randn(N, 3)
        rand_points /= np.linalg.norm(rand_points, axis=1, keepdims=True)
        starts.append(rand_points)
        
    # 3. HSS-like: Try to construct a configuration with some symmetry?
    # For now, random and Fibonacci are good.
    
    for start in starts:
        if time.time() - start_time > TIME_LIMIT - 10:
            break
            
        points, score = _optimize_from_start(start)
        
        if score < best_score:
            best_score = score
            best_points = points
            
    # Final normalization
    if best_points is None:
        best_points = fib_points
        
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
