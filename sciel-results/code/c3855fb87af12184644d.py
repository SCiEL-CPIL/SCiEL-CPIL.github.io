# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms

def compute_covering_radius(points):
    """
    Computes the covering radius (max circumradius of convex hull facets) in radians.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi # Invalid or degenerate, return large value
        
    simplices = hull.simplices
    max_radius = 0.0
    
    # Optimization: We only need the max radius.
    # Circumradius of a triangle on the unit sphere with vertices a,b,c:
    # The circumcenter is the normalized cross product of (b-a) and (c-a)? No.
    # The plane passing through a,b,c intersects the sphere. The center of the circular arc is the projection of the origin onto the plane.
    # Let n = (b-a) x (c-a). The normal to the plane.
    # The distance from origin to plane is h = |dot(a, n)| / |n|.
    # The radius of the circle is sqrt(1 - h^2).
    # The angular radius is arcsin(r_circ) or arccos(h) ?
    # If the origin is on the same side as the normal, h is positive.
    # The angular distance from the center (which is at vector n/|n| or -n/|n|) to the vertices is the circumradius.
    # Actually, the circumcenter of the spherical triangle is the point on the sphere in the direction of the normal to the plane, towards the triangle.
    # Let v = np.cross(b-a, c-a).
    # Normalize v to get the pole.
    # Check if the pole is on the same side as the vertices. dot(v, a) > 0?
    # If dot(v, a) < 0, flip v.
    # Then the circumradius is arccos(dot(v_norm, a)).
    
    for simplex in simplices:
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        
        v1 = b - a
        v2 = c - a
        normal = np.cross(v1, v2)
        norm_n = np.linalg.norm(normal)
        
        if norm_n < 1e-10:
            # Degenerate triangle, radius is pi/2 or something? 
            # If points are collinear on sphere, they form a great circle arc. 
            # ConvexHull shouldn't produce degenerate simplices for non-coplanar sets, 
            # but numerical errors might occur.
            # A degenerate hull facet implies the points are nearly coplanar with origin?
            # If they are coplanar with origin, they are on a great circle.
            # The "cap" covering them is a hemisphere? No.
            # If the normal is 0, the points are collinear.
            # This is rare. We can just skip or assign a high radius.
            max_radius = np.pi
            return max_radius
            
        pole = normal / norm_n
        
        # Ensure pole points towards the triangle
        if np.dot(pole, a) < 0:
            pole = -pole
            
        # The angular distance from pole to a is the circumradius
        cos_r = np.clip(np.dot(pole, a), -1.0, 1.0)
        r = np.arccos(cos_r)
        
        if r > max_radius:
            max_radius = r
            
    return max_radius

def construct():
    start_time = time.time()
    time_limit = 115.0 # Leave 5s buffer
    
    best_points = None
    best_radius = np.pi
    
    # Multi-restart Simulated Annealing
    # We will run multiple short SA runs and keep the best result.
    
    # Parameters
    T_start = 0.5
    T_end = 0.001
    cooling_rate = 0.995 # Adjust based on steps
    
    # We estimate steps based on time.
    # Let's do a quick benchmark of one hull computation.
    dummy_points = np.random.randn(N, 3)
    dummy_points = normalize(dummy_points)
    
    t0 = time.time()
    for _ in range(10):
        compute_covering_radius(dummy_points)
    t1 = time.time()
    avg_time = (t1 - t0) / 10.0
    
    # Estimate max iterations
    max_iter = int((time_limit - (time.time() - start_time)) / (avg_time * 1.5)) # 1.5x safety factor
    if max_iter < 1000:
        max_iter = 1000
    if max_iter > 50000:
        max_iter = 50000
        
    # Number of restarts
    # If we have 50000 iters, maybe 5 restarts of 10000?
    # Or just one long run? SA benefits from restarts to escape deep local minima.
    n_restarts = 5
    iters_per_restart = max_iter // n_restarts
    
    for r in range(n_restarts):
        # Initialize
        # Use random initialization for diversity
        points = np.random.randn(N, 3)
        points = normalize(points)
        
        # Initial evaluation
        curr_radius = compute_covering_radius(points)
        curr_points = points.copy()
        
        T = T_start
        
        for i in range(iters_per_restart):
            if time.time() - start_time > time_limit:
                break
                
            # Pick a random point to perturb
            idx = np.random.randint(N)
            
            # Perturb: add small random vector
            # Step size should be related to T
            step_size = T * 0.5
            perturb = np.random.randn(3) * step_size
            new_point = curr_points[idx] + perturb
            new_point = new_point / np.linalg.norm(new_point)
            
            # Create new points array
            new_points = curr_points.copy()
            new_points[idx] = new_point
            
            # Evaluate
            new_radius = compute_covering_radius(new_points)
            
            # Acceptance
            delta = new_radius - curr_radius
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                curr_points = new_points
                curr_radius = new_radius
                
                # Update best
                if curr_radius < best_radius:
                    best_radius = curr_radius
                    best_points = curr_points.copy()
            
            # Cool
            T *= cooling_rate
            if T < T_end:
                T = T_end
                
        # End of restart loop
        if time.time() - start_time > time_limit:
            break
            
    if best_points is None:
        # Fallback
        best_points = np.random.randn(N, 3)
        best_points = normalize(best_points)
        
    return best_points
# EVOLVE-BLOCK-END
