"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 130
DEADLINE = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START


def _covering_radius(points):
    """Compute the covering radius in degrees for a set of points on the unit sphere.
    
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest point in the configuration. This equals the largest circumradius
    of the facets of the convex hull.
    """
    from scipy.spatial import ConvexHull
    from scipy.spatial import Delaunay
    
    points = np.asarray(points)
    # Ensure points are on unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
    except Exception:
        # Fallback: use nearest-neighbor based estimate
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        # Sample points on sphere to estimate covering radius
        n_samples = 100000
        # Generate random points on sphere
        u = np.random.rand(n_samples, 3)
        u -= 0.5
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        _, idx = tree.query(u)
        dists = np.arccos(np.clip(np.sum(u * points[idx], axis=1), -1, 1))
        return np.degrees(np.max(dists))
    
    max_radius = 0.0
    for simplex in simplices:
        p0, p1, p2 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        # Compute circumradius of triangle on unit sphere
        # The circumcenter of a spherical triangle
        # For a triangle on the unit sphere, the circumradius is the angular distance
        # from the circumcenter to each vertex.
        
        # Method: find the point on the sphere equidistant from all three vertices
        # This is the intersection of the sphere with the plane through the three points
        # and the line perpendicular to that plane through the origin... no.
        
        # Better: The circumcenter of a spherical triangle can be found as follows:
        # The circumcenter C satisfies |C - P_i| = R for i=0,1,2 (on the sphere)
        # Equivalently, C · P_i = cos(R) for all i.
        # So C is in the null space of the system where C · (P_i - P_0) = 0
        # and C · P_0 = cos(R)
        
        # Actually, for a triangle inscribed in a unit sphere, the circumradius
        # can be computed from the side lengths.
        # Let a, b, c be the angular side lengths (opposite to vertices A, B, C)
        # Then cos(R) = cos(a)cos(b)cos(c) + sqrt((1-cos^2(a))(1-cos^2(b))(1-cos^2(c)))
        # Wait, that's not quite right. Let me use a different approach.
        
        # The circumcenter of a spherical triangle with vertices v1, v2, v3:
        # It lies on the line through the origin in the direction of the normal to the
        # plane containing v1, v2, v3... no, it's more subtle.
        
        # Alternative: The circumradius R satisfies:
        # For the triangle with vertices A, B, C on unit sphere,
        # the circumcenter O satisfies O·A = O·B = O·C = cos(R)
        # and |O| = 1 (O is on the unit sphere).
        # 
        # So O is a unit vector such that O·A = O·B = O·C.
        # This means O is parallel to A × B + B × C + C × A (the normal to the plane
        # containing A, B, C is not quite right either).
        
        # Let me just compute it numerically:
        # O = normalize(A × B + B × C + C × A) might not work directly.
        
        # Actually, the correct formula: the circumcenter of a spherical triangle
        # with vertices a, b, c (unit vectors) is proportional to:
        # (a × b) × (b × c) + (b × c) × (c × a) + (c × a) × (a × b)
        # Hmm, this is getting complicated. Let me use a simpler approach.
        
        # Compute the circumradius using the formula:
        # Given three points on the unit sphere, the circumradius R is given by:
        # sin(R) = abc / (4 * area) where a,b,c are chord lengths and area is the
        # area of the triangle in 3D space.
        
        # Actually, let me just use the fact that for three points on a unit sphere,
        # the circumradius can be found by:
        # The three points lie on a circle. The radius of that circle (in 3D) is r_3d.
        # The angular circumradius R satisfies: sin(R) = r_3d / 1 = r_3d
        # Wait no. If the circle has radius r_3d in 3D and is at distance d from origin,
        # then r_3d^2 + d^2 = 1 (since points are on unit sphere).
        # The angular distance from the center of the circle (projected to sphere) to
        # each vertex is R, and sin(R) = r_3d, cos(R) = d.
        
        # Compute the plane containing p0, p1, p2:
        n = np.cross(p1 - p0, p2 - p0)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            # Degenerate triangle
            continue
        n = n / n_norm
        
        # Distance from origin to the plane:
        d = np.dot(p0, n)
        
        # Radius of the circle in 3D:
        r_3d = np.sqrt(max(0.0, 1.0 - d * d))
        
        # Angular circumradius:
        if r_3d < 1e-12:
            R = 0.0
        else:
            R = np.arcsin(np.clip(r_3d, -1.0, 1.0))
        
        max_radius = max(max_radius, R)
    
    return np.degrees(max_radius)


def _hill_climb(points, max_time=None):
    """Perform hill-climbing optimization to minimize the covering radius."""
    points = points.copy()
    n = len(points)
    
    # Initial step size
    step = 0.01
    best_radius = _covering_radius(points)
    best_points = points.copy()
    
    start_time = time.time()
    
    while True:
        if max_time is not None and time.time() - start_time > max_time:
            break
        if time.time() > DEADLINE - 5:
            break
        
        improved = False
        for _ in range(50):  # Inner iterations per outer step
            if time.time() > DEADLINE - 5:
                break
            
            # Pick a random point to move
            i = np.random.randint(n)
            orig = points[i].copy()
            
            # Try moving in a random direction
            for _ in range(10):
                # Random direction on the sphere
                d = np.random.randn(3)
                d /= np.linalg.norm(d)
                
                # Move point i in direction d by step
                new_point = points[i] + step * d
                new_point /= np.linalg.norm(new_point)
                points[i] = new_point
                
                # Check if this improves the covering radius
                # Use a quick estimate: check distances to neighbors
                # For speed, only check if the moved point is closer to its nearest neighbor
                # or if it helps cover a gap
                # For now, use the full covering radius computation (slow but accurate)
                # Actually, let's use a faster heuristic:
                # Compute the maximum distance from the moved point to its k nearest neighbors
                # This is an upper bound on the local contribution
                
                # Quick check: compute distances to all other points
                dists = np.arccos(np.clip(np.sum(points * points[i], axis=1), -1, 1))
                # The covering radius is at least half the max gap
                # But this is expensive. Let's just do the full computation every few steps.
                
                if np.random.rand() < 0.1:  # Full check every 10% of the time
                    r = _covering_radius(points)
                    if r < best_radius - 1e-9:
                        best_radius = r
                        best_points = points.copy()
                        improved = True
                        break
                    else:
                        points[i] = orig
                        break
                else:
                    # Quick heuristic: accept if the point moved to a better position
                    # based on local density
                    pass
            
            if improved:
                # Reduce step size
                step *= 0.9
                break
        
        if not improved and step < 1e-6:
            break
        
        # Annealing: occasionally increase step size to escape local minima
        if np.random.rand() < 0.1:
            step = min(step * 1.5, 0.1)
    
    return best_points, best_radius


def _random_start():
    """Generate a random configuration of N points on the unit sphere."""
    # Use a more structured random start
    # Start with a Fibonacci spiral and add random perturbation
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Add random perturbation
    noise = np.random.randn(N, 3) * 0.1
    points += noise
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    return points


def _fibonacci_start():
    """Generate a Fibonacci spiral configuration."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    """Construct 130 points on the unit sphere minimizing the covering radius."""
    start_time = time.time()
    
    # Start with Fibonacci spiral
    best_points = _fibonacci_start()
    best_radius = _covering_radius(best_points)
    
    # Multiple random restarts with hill climbing
    n_restarts = 20
    for restart in range(n_restarts):
        if time.time() > DEADLINE - 10:
            break
        
        # Generate a random starting point
        if restart == 0:
            points = _fibonacci_start()
        else:
            points = _random_start()
        
        # Hill climb from this start
        remaining_time = DEADLINE - 10 - time.time()
        if remaining_time < 5:
            break
        
        # Allocate time based on restart number (more time for earlier restarts)
        time_per_restart = remaining_time / (n_restarts - restart)
        time_per_restart = min(time_per_restart, 5.0)  # Cap at 5 seconds per restart
        
        optimized_points, radius = _hill_climb(points, max_time=time_per_restart)
        
        if radius < best_radius:
            best_radius = radius
            best_points = optimized_points
    
    # Final check
    return best_points
# EVOLVE-BLOCK-END
