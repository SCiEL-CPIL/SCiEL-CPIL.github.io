"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

DEADLINE = 140.0
N = 50

def _covering_radius_rad(points):
    """
    Compute the covering radius (in radians) of points on the unit sphere.
    The covering radius is the maximum circumradius of the facets of the convex hull.
    """
    # Normalize points
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    points = points / norms
    
    # Check for coincident points
    dots = points @ points.T
    np.fill_diagonal(dots, 0)
    if np.max(dots) > 1 - 1e-8:
        return np.pi  # Invalid: coincident points
    
    # Check if all points are in a closed hemisphere
    # This is equivalent to checking if the origin is not in the convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi  # Degenerate
    
    # The origin is inside the convex hull if and only if the points are not in any closed hemisphere
    # We can check this by seeing if there exists a vector v such that v·p_i > 0 for all i
    # Alternatively, check if the convex hull contains the origin
    # A simpler check: if all points have positive dot product with some direction, they're in a hemisphere
    
    # Compute the normal for each facet and check if the origin is on the same side
    max_r = 0.0
    
    for simplex in hull.simplices:
        a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        idx_a, idx_b, idx_c = simplex
        
        # Compute normal to the plane (outward pointing)
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        
        n = n / n_norm
        
        # The outward normal should point away from the origin
        # Check: if n·a > 0, then the origin is on the opposite side (good, not in hemisphere)
        # If n·a < 0, flip n
        if np.dot(n, a) < 0:
            n = -n
        
        # The circumcenter is along the normal direction
        # Distance from origin to the plane = |n·a|
        dist_to_plane = abs(np.dot(n, a))
        
        # The circumradius of the spherical triangle
        # For a spherical triangle with vertices a, b, c, the circumradius R satisfies
        # cos(R) = (cos(ab) + cos(bc) + cos(ca) + 1) / (4 * cos(a) * cos(b) * cos(c)) ... no that's not right
        
        # Actually, for a spherical triangle, the circumradius can be computed as follows:
        # The circumcenter is the point on the sphere equidistant from all three vertices.
        # It lies along the normal to the plane of the triangle.
        
        # The circumcenter c satisfies:
        # c·a = c·b = c·c = cos(R)
        # c = t * n for some t (since c is along the normal)
        # t * n·a = cos(R), t * n·b = cos(R), t * n·c = cos(R)
        # So n·a = n·b = n·c = cos(R)/t
        
        # Let d_a = n·a, d_b = n·b, d_c = n·c
        # For a valid circumcenter, we need d_a = d_b = d_c (approximately)
        # The circumradius is then R = arccos(d_a / |c|) but |c| = 1
        
        # Actually, the circumcenter is at c = n / |n·a| if n·a > 0
        # Then R = arccos(n·a / |n·a|) = arccos(1) = 0? No, that's wrong.
        
        # Let me think again. The circumcenter is the point on the unit sphere that is
        # equidistant from a, b, c. It lies in the plane spanned by a, b, c, which is
        # the same as the plane perpendicular to n.
        
        # The circumcenter c satisfies:
        # c·a = c·b = c·c = cos(R)
        # c = αa + βb + γc (linear combination)
        # And |c| = 1
        
        # A simpler approach: the circumradius of a spherical triangle with side lengths
        # a, b, c (arcs) can be computed using the formula:
        # cos(R) = (cos(a)cos(b)cos(c) + 2cos(a)cos(b)cos(c) - cos²(a) - cos²(b) - cos²(c)) / (4cos(a)cos(b)cos(c))
        # No, that's not right either.
        
        # Let me use a different approach. The circumcenter of a spherical triangle is the
        # pole of the great circle through the three vertices. The circumradius is the
        # angular distance from the circumcenter to any vertex.
        
        # The circumcenter is along the normal n (or -n). Let's try both.
        for sign in [1, -1]:
            c_center = sign * n
            d_a = np.dot(c_center, a)
            d_b = np.dot(c_center, b)
            d_c = np.dot(c_center, c)
            
            # Check if this is a valid circumcenter (equidistant from all three vertices)
            if abs(d_a - d_b) < 1e-6 and abs(d_b - d_c) < 1e-6:
                # Check if the circumcenter is "empty" (no other points closer to the circumcenter)
                dots_all = points @ c_center
                # The circumcenter is valid if a, b, c are the closest points
                # (i.e., no other point is closer to the circumcenter)
                min_dots = np.min(dots_all)
                # The three vertices should have the largest dot product with the circumcenter
                # (smallest angular distance)
                if (d_a >= min_dots - 1e-9 and d_b >= min_dots - 1e-9 and d_c >= min_dots - 1e-9):
                    # Actually, we want the circumcenter to be the farthest from all points
                    # No, the covering radius is the maximum over all y of min_i ang_dist(y, p_i)
                    # The worst case is at the circumcenter of the largest empty circle
                    # So we want to find the circumcenter that is farthest from all points
                    
                    # The angular distance from the circumcenter to vertex a is arccos(d_a)
                    R = np.arccos(np.clip(d_a, -1.0, 1.0))
                    max_r = max(max_r, R)
    
    return max_r

def _fast_covering_radius(points):
    """
    Faster approximate covering radius for use in optimization.
    Uses a grid-based approach to estimate the maximum distance.
    """
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    points = points / norms
    
    # Generate a grid of test points on the sphere
    # Use a Fibonacci spiral for good coverage
    n_grid = 2000
    i = np.arange(n_grid) + 0.5
    z = 1.0 - 2.0 * i / n_grid
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute distances from grid points to all configuration points
    dots = grid @ points.T
    dots = np.clip(dots, -1.0, 1.0)
    min_dots = np.max(dots, axis=1)  # Maximum dot product = minimum distance
    max_dist = np.arccos(np.clip(min_dots, -1.0, 1.0))
    
    return np.max(max_dist)

def construct():
    start_time = time.time()
    
    # Start from Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _covering_radius_rad(best_points)
    
    # Phase 1: Gradient-based optimization using the fast radius
    def obj_fast(x):
        pts = x.reshape(-1, 3)
        return _fast_covering_radius(pts)
    
    try:
        result = minimize(
            obj_fast,
            points.flatten(),
            method='Powell',
            options={'maxiter': 100, 'maxfev': 3000}
        )
        opt_points = result.x.reshape(-1, 3)
        opt_radius = _covering_radius_rad(opt_points)
        if opt_radius < best_radius:
            best_points = opt_points
            best_radius = opt_radius
    except Exception:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 20:
        norms = np.linalg.norm(best_points, axis=1, keepdims=True)
        norms = np.where(norms < 1e-10, 1.0, norms)
        return best_points / norms
    
    # Phase 2: Simulated annealing with exact covering radius
    temp = 0.05
    cooling = 0.95
    best_sa_points = best_points.copy()
    best_sa_radius = best_radius
    current_points = best_points.copy()
    current_radius = best_radius
    
    n_sa_iters = 2000
    for it in range(n_sa_iters):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Propose a perturbation
        idx = np.random.randint(N)
        noise = np.random.randn(3) * temp
        new_points = current_points.copy()
        new_points[idx] = current_points[idx] + noise
        new_points[idx] = new_points[idx] / np.linalg.norm(new_points[idx])
        
        new_radius = _covering_radius_rad(new_points)
        
        # Accept if better or with probability exp(-Δ/T)
        delta = new_radius - current_radius
        if delta < 0 or np.random.rand() < np.exp(-delta / temp):
            current_points = new_points
            current_radius = new_radius
            if current_radius < best_sa_radius:
                best_sa_points = current_points.copy()
                best_sa_radius = current_radius
        
        temp *= cooling
    
    if best_sa_radius < best_radius:
        best_points = best_sa_points
        best_radius = best_sa_radius
    
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    best_points = best_points / norms
    
    return best_points
# EVOLVE-BLOCK-END
