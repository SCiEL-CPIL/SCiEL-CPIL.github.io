import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def compute_facet_data(points):
    """
    Computes the circumradius and circumcenter direction for each facet of the convex hull.
    Returns:
        radii: array of shape (M,)
        centers: array of shape (M, 3) unit vectors
        simplices: array of shape (M, 3)
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        return np.array([]), np.array([]), np.array([])
    
    if len(simplices) == 0:
        return np.array([]), np.array([]), np.array([])
        
    a = points[simplices[:, 0]]
    b = points[simplices[:, 1]]
    c = points[simplices[:, 2]]
    
    # Compute normals
    n = np.cross(b - a, c - a)
    norms = np.linalg.norm(n, axis=1)
    
    # Handle degenerate facets (should be rare for well-conditioned points)
    valid = norms > 1e-10
    if not np.all(valid):
        # If any are degenerate, we can't reliably compute centers for them.
        # We'll set their radius to pi (max) and center to (0,0,1) to be safe, 
        # or just skip them if we want to be strict. 
        # For optimization, treating them as "bad" encourages moving away from degeneracy.
        pass
        
    # Normalize normals
    u = np.zeros_like(n)
    u[valid] = n[valid] / norms[valid, np.newaxis]
    u[~valid] = [0, 0, 1]
    
    # Orient normals outwards (away from origin)
    centroids = (a + b + c) / 3.0
    dots = np.sum(u * centroids, axis=1)
    u[dots < 0] *= -1
    
    # Circumradius
    # The circumcenter of a triangle on the unit sphere is the intersection of the plane
    # of the triangle with the line through the origin perpendicular to the plane.
    # The distance from origin to the plane is 1/|u| if u is not normalized? 
    # Wait, u is the unit normal. The plane is u . x = d.
    # Since a, b, c are on the unit sphere, d = u . a.
    # The circumcenter on the sphere is the point y on the sphere such that y is in the plane 
    # and y is the projection of the origin onto the plane? No.
    # The circumcenter of the spherical triangle is the intersection of the perpendicular bisectors.
    # For a triangle inscribed in the unit sphere, the circumcenter (on the sphere) is 
    # parallel to the vector n = (b-a) x (c-a).
    # Actually, the vector u (unit normal) points to the circumcenter.
    # Let's verify: The circumcenter C satisfies |C-a|^2 = |C-b|^2 = |C-c|^2 = 2R^2.
    # Since C is on the sphere, |C|=1.
    # C is in the direction of u. So C = u.
    # Then |u-a|^2 = |u|^2 + |a|^2 - 2u.a = 2 - 2cos(theta).
    # So R = arccos(cos(theta)) = arccos(u.a).
    # This is correct.
    
    cos_theta = np.sum(u * a, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    radii = np.arccos(cos_theta)
    
    # For degenerate facets, the radius calculation might be invalid.
    # We set radii for invalid normals to pi (worst case) to force movement.
    radii[~valid] = np.pi
    
    return radii, u, simplices

def greedy_init(n_points, seed=42):
    """
    Greedy initialization: start with a random point, then add points that maximize
    the minimum distance to existing points.
    """
    np.random.seed(seed)
    points = [np.random.randn(3)]
    points[0] /= np.linalg.norm(points[0])
    
    for _ in range(n_points - 1):
        # Generate many candidates
        candidates = np.random.randn(1000, 3)
        candidates /= np.linalg.norm(candidates, axis=1, keepdims=True)
        
        # For each candidate, find min distance to existing points
        # This is O(K * M * D)
        # We can vectorize: dists[i, j] = |c_i - p_j|^2
        # But 1000 * 130 * 3 is small.
        
        min_dists = np.full(len(candidates), np.inf)
        for p in points:
            d = np.linalg.norm(candidates - p, axis=1)
            min_dists = np.minimum(min_dists, d)
        
        # Pick the candidate with the largest min_dist
        best_idx = np.argmax(min_dists)
        points.append(candidates[best_idx])
        
    return np.array(points)

def construct():
    start_time = time.time()
    time_limit = 115.0 # Safety margin
    
    best_points = None
    best_radius = np.pi
    
    # Initial configurations
    # 1. Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    init_fib = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Greedy Initialization
    init_greedy = greedy_init(N, seed=10)
    
    # 3. Random
    np.random.seed(42)
    init_rand = np.random.randn(N, 3)
    init_rand /= np.linalg.norm(init_rand, axis=1, keepdims=True)
    
    initializations = [init_fib, init_greedy, init_rand]
    
    # We will run optimization on these, plus some perturbed versions if time permits.
    # The greedy init is often very strong.
    
    max_restarts = 10
    
    for restart in range(max_restarts):
        if time.time() - start_time > time_limit:
            break
            
        if restart < len(initializations):
            points = initializations[restart].copy()
        else:
            # Perturb the best found so far or the greedy init
            if best_points is not None and restart % 2 == 0:
                points = best_points + np.random.randn(N, 3) * 0.1
            else:
                points = init_greedy + np.random.randn(N, 3) * 0.2
            points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Optimization Parameters
        # T is the step size scale
        T = 0.05 
        cool_rate = 0.995
        no_improve_count = 0
        current_radius = np.pi
        
        # We run a fixed number of steps or until time runs out
        # ConvexHull takes ~1-2ms. 115s / 2ms = 57500 steps.
        # But we have overhead. Let's just loop until time limit.
        
        while True:
            if time.time() - start_time > time_limit:
                break
                
            radii, centers, simplices = compute_facet_data(points)
            
            if len(radii) == 0:
                break
                
            max_radius = np.max(radii)
            
            if max_radius < best_radius:
                best_radius = max_radius
                best_points = points.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                # If stuck, cool down faster
                if no_improve_count > 20:
                    T *= 0.95
                if no_improve_count > 100:
                    break
                if T < 1e-5:
                    break
            
            # Identify facets to move: Top 20%
            if len(radii) > 1:
                threshold = np.percentile(radii, 80)
                mask = radii > threshold
            else:
                mask = np.ones(len(radii), dtype=bool)
            
            if not np.any(mask):
                # If no facets are "bad" (shouldn't happen unless all equal), pick the largest
                max_idx = np.argmax(radii)
                mask = np.zeros(len(radii), dtype=bool)
                mask[max_idx] = True
            
            # Accumulate updates
            updates = np.zeros_like(points)
            
            # Iterate over bad facets
            # This loop is over ~2N facets, so ~260 iterations. Very fast.
            for m in np.where(mask)[0]:
                idx = simplices[m]
                i_idx, j_idx, k_idx = idx
                center = centers[m]
                radius = radii[m]
                
                # Move towards center
                # Step size proportional to radius and T
                step_size = T * radius
                
                updates[i_idx] += step_size * (center - points[i_idx])
                updates[j_idx] += step_size * (center - points[j_idx])
                updates[k_idx] += step_size * (center - points[k_idx])
            
            points += updates
            # Normalize back to sphere
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms < 1e-10] = 1.0
            points /= norms
            
            T *= cool_rate

    if best_points is None:
        best_points = init_fib
        
    return best_points
