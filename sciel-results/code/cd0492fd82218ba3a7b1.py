"""Rewritten program for covering the sphere with 50 equal caps."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import warnings

def _angular_distance(p, q):
    """Angular distance in radians between two unit vectors."""
    dot = np.clip(np.dot(p, q), -1.0, 1.0)
    return np.arccos(dot)

def _covering_radius(points):
    """Compute the covering radius in degrees for a set of points on the unit sphere.
    
    The covering radius is the maximum angular distance from any point on the sphere
    to its nearest configuration point. This equals the largest circumradius of the
    facets of the convex hull (the spherical caps).
    """
    points = np.array(points)
    n = len(points)
    
    # Normalize to be safe
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    points = points / norms
    
    # Check for coincident points
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(points[i] - points[j]) < 1e-12:
                return 180.0  # Invalid
    
    # Check if all points are in a closed hemisphere
    # A set of points is in a closed hemisphere if there exists a vector v such that
    # v · p_i >= 0 for all i. We can check this via linear programming or by checking
    # if the origin is inside the convex hull.
    # If the origin is NOT in the interior of the convex hull, the points are in a hemisphere.
    try:
        hull = ConvexHull(points)
    except Exception:
        return 180.0
    
    # Check if origin is inside the convex hull
    # The origin is inside if for all facets, the distance from origin to facet plane > 0
    # and the origin is on the "inside" side.
    # For a convex hull of points on the unit sphere, the origin is inside iff the hull
    # contains the origin in its interior.
    
    # Simple check: the origin is inside the convex hull iff the minimum distance
    # from the origin to any facet is positive and the origin is on the same side
    # as the interior.
    
    # Actually, for points on the unit sphere, the origin is inside the convex hull
    # if and only if the points are NOT contained in any closed hemisphere.
    
    # We can check this by solving: does there exist v such that v·p_i >= 0 for all i?
    # This is equivalent to checking if the convex hull of the points contains the origin
    # in its interior (or boundary).
    
    # Alternative: check if the spherical convex hull covers the whole sphere.
    # The covering radius is half the largest empty cap.
    
    # Compute the covering radius via the convex hull facets.
    # For each facet (triangle on the convex hull), the corresponding spherical cap
    # has its center at the normalized facet normal, and the radius is the angular
    # distance from the center to the farthest vertex.
    
    max_radius = 0.0
    for simplex in hull.simplices:
        # Get the three points of this facet
        p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        # The facet normal (outward)
        normal = np.cross(p2 - p1, p3 - p1)
        norm_n = np.linalg.norm(normal)
        if norm_n < 1e-15:
            continue
        normal = normal / norm_n
        
        # The center of the spherical cap is at the normalized normal
        center = normal
        
        # The angular radius is the maximum angular distance from center to any vertex
        # of this facet
        for p in [p1, p2, p3]:
            d = _angular_distance(center, p)
            if d > max_radius:
                max_radius = d
        
        # Check that the cap is empty: no other point is closer to the center
        # than the vertices are (i.e., the cap doesn't contain any other point)
        # Actually, for the covering radius, we need the largest empty spherical cap.
        # The cap centered at `center` with radius `max_radius` for this facet
        # should not contain any other configuration point in its interior.
        
    # But we also need to consider that the covering radius might be determined
    # by a different structure. Let's use a more robust approach.
    
    # The covering radius is the minimum over all possible centers c on the sphere
    # of the maximum angular distance from c to any configuration point... no wait.
    # The covering radius is the smallest r such that every point on the sphere is
    # within angular distance r of some configuration point.
    
    # Equivalently, it's the largest r such that there exists a point on the sphere
    # at angular distance r from all configuration points (or at least the nearest one
    # is at distance r).
    
    # The worst-case point is at the center of the largest empty spherical cap.
    # The centers of the largest empty caps are at the circumcenters of the facets
    # of the convex hull (for the Voronoi diagram on the sphere).
    
    # So the covering radius is the maximum over all facets of the angular distance
    # from the facet's circumcenter (on the sphere) to the farthest vertex.
    
    # Let me recompute more carefully.
    
    max_cov_radius = 0.0
    for simplex in hull.simplices:
        p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        # Compute the circumcenter of the spherical triangle
        # This is the point on the sphere equidistant from p1, p2, p3
        # It's the normalized cross product of (p2-p1) and (p3-p1), but that gives
        # the normal to the plane, not the circumcenter.
        
        # The circumcenter of a spherical triangle is the point where the perpendicular
        # bisectors of the sides meet. For a triangle on the unit sphere, the circumcenter
        # is in the direction of the cross product of the two edge vectors, but adjusted.
        
        # Actually, the circumcenter is the normalized vector:
        # c = (p1 + p2 + p3) normalized? No, that's the centroid.
        
        # The circumcenter of the spherical triangle with vertices p1, p2, p3 is:
        # c = normalize(cross(p2-p1, p3-p1)) only if the triangle is right-angled at p1.
        
        # More generally, the circumcenter is the point equidistant from all three vertices.
        # Let c = a*p1 + b*p2 + c*p3. We need c·p1 = c·p2 = c·p3 and ||c|| = 1.
        
        # This is equivalent to finding the intersection of the plane perpendicular
        # to (p2-p1) through the midpoint (p1+p2)/2, etc.
        
        # A simpler approach: the circumcenter is the normalized vector
        # c = p1 × (p2 × p3) + ... hmm, this is getting complicated.
        
        # Let's use the fact that for a triangle on the unit sphere, the circumcenter
        # is in the direction of:
        # c = (p1 × p2) × (p2 × p3) + ... 
        
        # Actually, the circumcenter of a spherical triangle is given by:
        # c = normalize(cross(cross(p1, p2), cross(p2, p3)))
        # Wait, that's not right either.
        
        # Let me just compute it numerically. The circumcenter c satisfies:
        # c·p1 = c·p2 = c·p3 = cos(r) for some r.
        # So c is in the null space of [p1-p2; p1-p3] (in the sense that c·(p1-p2)=0 and c·(p1-p3)=0).
        # So c is parallel to cross(p1-p2, p1-p3).
        
        c_dir = np.cross(p1 - p2, p1 - p3)
        c_norm = np.linalg.norm(c_dir)
        if c_norm < 1e-15:
            continue
        c_dir = c_dir / c_norm
        
        # Make sure c_dir is on the same side as the outward normal
        normal = np.cross(p2 - p1, p3 - p1)
        if np.dot(c_dir, normal) < 0:
            c_dir = -c_dir
        
        # The circumradius is the angular distance from c_dir to any vertex
        r = _angular_distance(c_dir, p1)
        
        # Verify that no other point is inside this cap
        # i.e., for all other points p_j, _angular_distance(c_dir, p_j) >= r
        # If some point is closer, then this isn't an empty cap.
        is_empty = True
        for j in range(n):
            if j not in simplex:
                d = _angular_distance(c_dir, points[j])
                if d < r - 1e-10:
                    is_empty = False
                    break
        
        if is_empty and r > max_cov_radius:
            max_cov_radius = r
    
    # Convert to degrees
    return np.degrees(max_cov_radius)


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    # Start with a good initial configuration: icosahedral symmetry based
    # or use a known good starting point.
    
    # Let's start with a Fibonacci spiral and then optimize.
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Optimize using gradient-free methods to minimize the covering radius
    # The covering radius is a piecewise smooth function, so we use Nelder-Mead
    # or similar.
    
    # We'll use a multi-start approach with local optimization.
    
    def objective(x):
        """Objective function: covering radius in degrees."""
        pts = x.reshape(N, 3)
        # Normalize
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        pts = pts / norms
        
        # Check for coincident points
        for i in range(N):
            for j in range(i + 1, N):
                if np.linalg.norm(pts[i] - pts[j]) < 1e-10:
                    return 180.0
        
        return _covering_radius(pts)
    
    # Try multiple random starts and keep the best
    best_points = points.copy()
    best_val = objective(points.flatten())
    
    rng = np.random.RandomState(42)
    
    for start in range(5):
        if start == 0:
            x0 = points.flatten()
        else:
            # Random perturbation of the initial configuration
            x0 = points.flatten() + rng.randn(N * 3) * 0.1
        
        # Normalize
        x0 = x0.reshape(N, 3)
        norms = np.linalg.norm(x0, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        x0 = (x0 / norms).flatten()
        
        # Use Nelder-Mead
        try:
            result = minimize(objective, x0, method='Nelder-Mead', 
                             options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            if result.fun < best_val:
                best_val = result.fun
                best_points = result.x.reshape(N, 3)
                norms = np.linalg.norm(best_points, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                best_points = best_points / norms
        except Exception:
            pass
    
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    best_points = best_points / norms
    
    return best_points
# EVOLVE-BLOCK-END
