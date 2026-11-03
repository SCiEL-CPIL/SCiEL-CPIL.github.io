import numpy as np
import scipy.optimize as opt
import time
import scipy.spatial as sp

N = 50

# Precompute sample points for fast objective evaluation
# We use a Fibonacci spiral for uniform distribution
def generate_sample_points(num_samples):
    i = np.arange(num_samples) + 0.5
    z = 1.0 - 2.0 * i / num_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

SAMPLES = generate_sample_points(20000)

def normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return points / norms

def covering_radius_sampled(points):
    """
    Approximate covering radius using precomputed samples.
    points: (N, 3)
    """
    points = normalize(points)
    # Compute dot products: (num_samples, N)
    dots = SAMPLES @ points.T
    # Clip to [-1, 1] for numerical stability
    dots = np.clip(dots, -1.0, 1.0)
    # Min distance to any point
    min_dots = dots.max(axis=1)  # Max dot product corresponds to min angle
    # Angles
    angles = np.arccos(min_dots)
    return angles.max()

def covering_radius_exact(points):
    """
    Exact covering radius using convex hull circumradii.
    points: (N, 3)
    """
    points = normalize(points)
    hull = sp.ConvexHull(points)
    max_r = 0.0
    for simplex in hull.simplices:
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        
        # Circumcenter of the triangle on the sphere
        # The circumcenter is the intersection of the sphere with the line passing through the origin
        # and perpendicular to the plane of the triangle? No.
        # The circumcenter of the spherical triangle is the point on the sphere equidistant from the 3 vertices.
        # In 3D, the plane of the triangle has a normal vector n = (p2-p1) x (p3-p1).
        # The circumcenter c is in the direction of the projection of the origin onto the plane?
        # Actually, for a triangle on the unit sphere, the circumcenter is the normalized vector 
        # c = (p1 + p2 + p3) is NOT generally correct.
        
        # Correct method:
        # The circumcenter lies on the line through the origin and the centroid of the triangle? No.
        # The circumcenter is the intersection of the sphere and the line defined by the normal to the plane 
        # passing through the circumcenter of the planar triangle?
        
        # Simpler: The circumcenter c satisfies |c - p1| = |c - p2| = |c - p3| and |c|=1.
        # This implies c is in the direction of the vector sum of the unit vectors? No.
        # Let's use the formula for the circumcenter of a spherical triangle.
        # It is the normalized cross product of the edges? No.
        
        # The circumcenter is the normalized vector:
        # c = normalize( (p2 x p3) + (p3 x p1) + (p1 x p2) ) ? 
        # Let's verify.
        # If p1, p2, p3 are on the sphere, the plane equation is n . x = d.
        # The circumcenter is the intersection of the sphere and the line through the origin in the direction 
        # of the vector perpendicular to the plane? No, the circumcenter is the point on the sphere 
        # whose projection onto the plane is the circumcenter of the planar triangle.
        
        # Alternative robust method:
        # Solve for c = a*p1 + b*p2 + c*p3 such that |c|=1 and c.p1 = c.p2 = c.p3.
        # c.p1 = c.p2 => c.(p1-p2) = 0.
        # c.p2 = c.p3 => c.(p2-p3) = 0.
        # So c is parallel to (p1-p2) x (p2-p3).
        # Let v = (p1-p2) x (p2-p3).
        # c = normalize(v).
        # Check: c is perpendicular to p1-p2 and p2-p3. So c.p1 = c.p2 and c.p2 = c.p3.
        # So c is equidistant from p1, p2, p3.
        # Since c is on the sphere, this is the circumcenter.
        
        v = np.cross(p1 - p2, p2 - p3)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-10:
            continue
        c = v / norm_v
        
        # Distance from c to p1
        dot = np.dot(c, p1)
        dot = np.clip(dot, -1.0, 1.0)
        r = np.arccos(dot)
        if r > max_r:
            max_r = r
            
    return max_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    best_points = None
    best_score = np.inf
    
    # Initial configurations
    inits = []
    
    # 1. Fibonacci
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    inits.append(fib)
    
    # 2. Random
    for _ in range(5):
        rand = np.random.randn(N, 3)
        rand = rand / np.linalg.norm(rand, axis=1, keepdims=True)
        inits.append(rand)
        
    # 3. Grid (approximate)
    # Use spherical coordinates grid
    # This is often a bad start for covering, but worth trying
    # We'll skip complex grids and stick to random + fib
    
    for init in inits:
        if time.time() > deadline:
            break
            
        # Optimize using Powell on the sampled objective
        # Powell is robust to non-smoothness
        try:
            res = opt.minimize(
                covering_radius_sampled, 
                init, 
                method='Powell', 
                options={'maxiter': 1000, 'disp': False}
            )
            opt_points = normalize(res.x)
            opt_score = covering_radius_exact(opt_points)
            
            if opt_score < best_score:
                best_score = opt_score
                best_points = opt_points
        except Exception as e:
            pass
            
    # Polish the best point with a few steps of exact objective?
    # Powell on exact objective might be slow, but let's try a short run
    if best_points is not None and time.time() < deadline - 5:
        try:
            res = opt.minimize(
                covering_radius_exact, 
                best_points, 
                method='Nelder-Mead', 
                options={'maxiter': 200, 'disp': False, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            polish_points = normalize(res.x)
            polish_score = covering_radius_exact(polish_points)
            if polish_score < best_score:
                best_points = polish_points
                best_score = polish_score
        except Exception:
            pass

    return best_points

# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize as opt
import scipy.spatial as sp
import time

N = 50

def _generate_sample_points(num_samples):
    i = np.arange(num_samples) + 0.5
    z = 1.0 - 2.0 * i / num_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

_SAMPLE_POINTS = _generate_sample_points(20000)

def _normalize(points):
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return points / norms

def _covering_radius_sampled(points):
    points = _normalize(points)
    dots = _SAMPLE_POINTS @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    min_dots = dots.max(axis=1)
    angles = np.arccos(min_dots)
    return angles.max()

def _covering_radius_exact(points):
    points = _normalize(points)
    try:
        hull = sp.ConvexHull(points)
    except Exception:
        return np.pi # Invalid or degenerate, return large value
    max_r = 0.0
    for simplex in hull.simplices:
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]
        
        v = np.cross(p1 - p2, p2 - p3)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-10:
            continue
        c = v / norm_v
        
        dot = np.dot(c, p1)
        dot = np.clip(dot, -1.0, 1.0)
        r = np.arccos(dot)
        if r > max_r:
            max_r = r
    return max_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    best_points = None
    best_score = np.inf
    
    # 1. Fibonacci
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    inits = [fib]
    
    # 2. Random restarts
    for _ in range(10):
        rand = np.random.randn(N, 3)
        rand = _normalize(rand)
        inits.append(rand)
        
    for init in inits:
        if time.time() > deadline:
            break
            
        try:
            res = opt.minimize(
                _covering_radius_sampled, 
                init, 
                method='Powell', 
                options={'maxiter': 2000, 'disp': False}
            )
            opt_points = _normalize(res.x)
            opt_score = _covering_radius_exact(opt_points)
            
            if opt_score < best_score:
                best_score = opt_score
                best_points = opt_points
        except Exception:
            pass
            
    # Polish
    if best_points is not None and time.time() < deadline - 5:
        try:
            res = opt.minimize(
                _covering_radius_exact, 
                best_points, 
                method='Nelder-Mead', 
                options={'maxiter': 500, 'disp': False, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            polish_points = _normalize(res.x)
            polish_score = _covering_radius_exact(polish_points)
            if polish_score < best_score:
                best_points = polish_points
        except Exception:
            pass

    return best_points
# EVOLVE-BLOCK-END
