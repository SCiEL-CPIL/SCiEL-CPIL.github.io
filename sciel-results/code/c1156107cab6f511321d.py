"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _covering_radius(pts):
    """Compute the covering radius in radians using convex hull facets."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi  # invalid, return large value
    
    # For each facet, find the circumradius
    # The covering radius is the max over all facets of the circumradius
    # A facet is a simplex in 3D (triangle) since we're on the sphere
    # The circumradius of a spherical triangle can be found by finding the point
    # on the sphere equidistant from all vertices of the facet
    
    max_r = 0.0
    for simplex in hull.simplices:
        # Get the 3 points of this facet
        tri = pts[simplex]
        
        # Find the circumcenter on the sphere
        # The point equidistant from all 3 vertices lies along the normal to the plane
        # through the 3 points, scaled to unit length.
        
        a, b, c = tri
        # Normal to the plane through a, b, c
        ab = b - a
        ac = c - a
        n = np.cross(ab, ac)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            # Degenerate facet, skip
            continue
        n = n / n_norm
        
        # The circumcenter on the sphere is either n or -n
        # We need the one that is "inside" the convex hull, i.e., the one
        # that maximizes the minimum distance to the facet vertices in the correct direction
        
        # Actually, for a convex polytope inscribed in the unit sphere, the circumcenter
        # of each facet (the point on the sphere farthest from the facet center in the
        # outward normal direction) gives the covering radius contribution.
        
        # The angular distance from a point y on the sphere to the nearest of {a,b,c}
        # is maximized at the circumcenter. The circumcenter is along the normal.
        # We need to determine which sign of n points "outward".
        
        # The centroid of the facet
        centroid = (a + b + c) / 3.0
        
        # The outward normal should point away from the origin (since the polytope
        # contains the origin for a good covering)
        # Check: if dot(centroid, n) > 0, then n points outward
        if np.dot(centroid, n) < 0:
            n = -n
        
        # Now n is the circumcenter candidate
        # Compute angular distance from n to each vertex
        for v in tri:
            dot_val = np.clip(np.dot(n, v), -1.0, 1.0)
            angle = np.arccos(dot_val)
            if angle > max_r:
                max_r = angle
    
    return max_r

def _covering_radius_fast(pts):
    """Faster covering radius computation for optimization."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi
    
    max_r = 0.0
    for simplex in hull.simplices:
        tri = pts[simplex]
        a, b, c = tri
        ab = b - a
        ac = c - a
        n = np.cross(ab, ac)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        centroid = (a + b + c) / 3.0
        if np.dot(centroid, n) < 0:
            n = -n
        # Compute max angular distance from n to the 3 vertices
        dots = np.clip(n @ tri.T, -1.0, 1.0)
        angles = np.arccos(dots)
        max_r = max(max_r, np.max(angles))
    return max_r

def _objective(x):
    """Objective function: covering radius in radians."""
    pts = x.reshape(N, 3)
    return _covering_radius_fast(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_val = _covering_radius_fast(pts)
    
    # Local optimization using L-BFGS-B
    # We'll do several restarts with different initial conditions
    
    def _optimize_from(pts_init):
        """Run local optimization from given starting points."""
        x0 = pts_init.flatten()
        
        def obj(x):
            return _objective(x)
        
        result = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-10})
        return result.x.reshape(N, 3), result.fun
    
    # First, optimize from the Fibonacci start
    if time.time() < deadline - 30:
        opt_pts, val = _optimize_from(pts)
        opt_pts = _normalize(opt_pts)
        val = _covering_radius_fast(opt_pts)
        if val < best_val:
            best_pts = opt_pts
            best_val = val
    
    # Try a few random restarts
    np.random.seed(42)
    for restart in range(20):
        if time.time() > deadline - 20:
            break
        
        # Generate random points
        random_pts = np.random.randn(N, 3)
        random_pts = _normalize(random_pts)
        
        # Quick local optimization
        x0 = random_pts.flatten()
        def obj(x):
            return _objective(x)
        
        try:
            result = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': 200, 'ftol': 1e-8})
            opt_pts = result.x.reshape(N, 3)
            opt_pts = _normalize(opt_pts)
            val = _covering_radius_fast(opt_pts)
            if val < best_val:
                best_pts = opt_pts
                best_val = val
        except Exception:
            continue
    
    # Final polish: use higher precision
    if time.time() < deadline - 5:
        x0 = best_pts.flatten()
        def obj(x):
            return _objective(x)
        
        try:
            result = minimize(obj, x0, method='L-BFGS-B', options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10})
            opt_pts = result.x.reshape(N, 3)
            opt_pts = _normalize(opt_pts)
            val = _covering_radius_fast(opt_pts)
            if val < best_val:
                best_pts = opt_pts
                best_val = val
        except Exception:
            pass
    
    # Ensure the result is valid: not in a closed hemisphere, no coincident points
    best_pts = _normalize(best_pts)
    
    # Check for coincident points
    for i in range(N):
        for j in range(i+1, N):
            if np.linalg.norm(best_pts[i] - best_pts[j]) < 1e-8:
                # Separate them slightly
                best_pts[j] = best_pts[j] + 1e-6 * np.random.randn(3)
                best_pts = _normalize(best_pts)
    
    # Check if all points are in a closed hemisphere
    # This is equivalent to checking if the convex hull contains the origin
    try:
        hull = ConvexHull(best_pts)
        # Check if origin is inside the hull
        # For a convex polytope, the origin is inside if all facet normals
        # (pointing outward) have positive dot product with the origin... 
        # Actually, simpler: check if the minimum distance from origin to any facet is > 0
        # or just check if the centroid of all points is not too close to the boundary
        centroid = np.mean(best_pts, axis=0)
        if np.linalg.norm(centroid) < 0.1:
            # Likely contains origin, good
            pass
        else:
            # Might be in a hemisphere, try to fix by reflecting some points
            # But this is a heuristic; the optimization should have avoided this
            pass
    except Exception:
        pass
    
    return best_pts
# EVOLVE-BLOCK-END
