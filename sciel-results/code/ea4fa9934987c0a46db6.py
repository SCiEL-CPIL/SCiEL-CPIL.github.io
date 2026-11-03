"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def _covering_radius(points):
    """Compute the covering radius in radians using the convex hull.
    
    The covering radius is the maximum over all facets of the convex hull
    of the circumradius of that facet (the angular distance from the facet's
    circumcenter to its vertices). We also need to check that the circumcenter
    lies on the same side of the facet as the origin (empty cap condition).
    """
    points = np.asarray(points)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: compute max nearest-neighbor distance
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        # Sample points on sphere
        n_samples = 5000
        rng = np.random.RandomState(42)
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dists, _ = tree.query(samples)
        return np.min(dists)
    
    max_r = 0.0
    for simplex in hull.simplices:
        pts = points[simplex]
        # Compute circumcenter of the simplex on the sphere
        # For a triangle on the unit sphere, the circumcenter is the normalized cross product approach
        # Actually, for a spherical triangle, the circumcenter is the point equidistant from all three vertices
        # We can find it by intersecting the great circles perpendicular bisectors
        
        # For a triangle with vertices a, b, c on unit sphere:
        # The circumcenter x satisfies |x-a| = |x-b| = |x-c|, i.e., x·a = x·b = x·c
        # So x is orthogonal to (a-b) and (a-c), meaning x is parallel to (a-b)×(a-c)
        # But we need the correct sign (the one on the same side as the origin or opposite)
        
        # For a triangle: circumcenter direction = normalize((a-b) x (a-c)) or its negative
        # The angular radius is arccos(x·a) where x is the circumcenter
        
        a, b, c = pts[0], pts[1], pts[2]
        # Cross product gives a vector perpendicular to the plane of the triangle
        # The circumcenter on the sphere is along this direction (or opposite)
        v = np.cross(b - a, c - a)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            continue
        v = v / v_norm
        
        # The circumcenter is either v or -v. We pick the one that is "inside" the cap
        # i.e., the one for which the angular distance to vertices is < pi/2 (or the smaller one)
        dot_a = np.dot(v, a)
        # If dot_a is negative, the circumcenter is on the opposite side
        # The actual circumcenter should give the minimum angular distance
        # For a proper facet, the circumcenter is the one where x·a = x·b = x·c
        # and this value determines the radius
        
        # The circumradius in angle is arccos(|x·a|) if we take the correct sign
        # Actually, for the covering radius, we want the cap that contains no other points
        # The circumcenter of the facet should be the center of the empty cap
        # x·a should be positive (cap less than hemisphere) for valid facets
        
        cos_r = abs(dot_a)
        cos_r = np.clip(cos_r, -1.0, 1.0)
        r = np.arccos(cos_r)
        
        # Check empty cap: the circumcenter should not have any point closer than the vertices
        # i.e., for all points p, x·p <= x·a (the cap is empty)
        dots = points @ v
        # The cap centered at v with angular radius r contains all p with v·p >= cos_r
        # We need to verify no other point is strictly inside (this is automatically true for hull facets)
        # But we should check that v is on the correct side
        
        if dot_a < 0:
            # Flip
            v = -v
            cos_r = abs(dot_a)
            cos_r = np.clip(cos_r, -1.0, 1.0)
            r = np.arccos(cos_r)
        
        max_r = max(max_r, r)
    
    return max_r


def _fibonacci_points(n):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _local_optimize(points, max_iter=5000, lr_init=0.01, lr_min=1e-6, tol=1e-10):
    """Local optimization using gradient descent on the sphere.
    
    We minimize the covering radius by pushing points away from the worst facet.
    """
    points = np.array(points, dtype=float)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    lr = lr_init
    for it in range(max_iter):
        # Compute covering radius and identify worst facet
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # Find the facet with maximum circumradius
        worst_simplex = None
        worst_r = 0.0
        worst_center = None
        
        for simplex in hull.simplices:
            pts = points[simplex]
            a, b, c = pts[0], pts[1], pts[2]
            v = np.cross(b - a, c - a)
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-12:
                continue
            v = v / v_norm
            dot_a = np.dot(v, a)
            if dot_a < 0:
                v = -v
                dot_a = -dot_a
            cos_r = np.clip(abs(dot_a), -1.0, 1.0)
            r = np.arccos(cos_r)
            if r > worst_r:
                worst_r = r
                worst_simplex = simplex
                worst_center = v
        
        if worst_simplex is None:
            break
        
        # Move the vertices of the worst facet away from the circumcenter
        # The gradient of the covering radius w.r.t. each vertex points away from the circumcenter
        # projected onto the tangent plane at that vertex
        step = np.zeros_like(points)
        for idx in worst_simplex:
            p = points[idx]
            # Direction to move: away from circumcenter, projected to tangent plane
            d = worst_center - p
            d_tangent = d - np.dot(d, p) * p
            step[idx] += d_tangent
        
        # Also consider moving other points to fill gaps
        # Compute nearest neighbor distances for all points
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        dists, indices = tree.query(points, k=2)
        nn_dists = dists[:, 1]
        
        # Points with large nearest-neighbor distances should move toward the circumcenter
        # Actually, we want to reduce the covering radius, so we move points in the worst facet outward
        # and potentially move nearby points to balance
        
        step /= (np.linalg.norm(step) + 1e-12)
        
        new_points = points + lr * step
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        new_cr = _covering_radius(new_points)
        if new_cr < best_cr - tol:
            best_cr = new_cr
            best_points = new_points.copy()
            points = new_points
            lr = max(lr * 0.99, lr_min)
        else:
            lr = max(lr * 0.95, lr_min)
            if lr < lr_min * 1.01:
                break
    
    return best_points, best_cr


def _simulated_annealing(points, max_iter=3000, t_init=0.1, t_min=1e-4):
    """Simulated annealing to escape local minima."""
    points = np.array(points, dtype=float)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    best_points = points.copy()
    best_cr = _covering_radius(points)
    current_cr = best_cr
    
    t = t_init
    rng = np.random.RandomState(123)
    
    for it in range(max_iter):
        # Random perturbation
        idx = rng.randint(N)
        angle = rng.uniform(0, 2 * np.pi)
        axis = rng.randn(3)
        axis /= np.linalg.norm(axis)
        
        # Rotate point idx by small angle around random axis
        theta = rng.uniform(-t, t)
        # Rodrigues rotation
        c, s = np.cos(theta), np.sin(theta)
        k = axis
        p = points[idx]
        new_p = p * c + np.cross(k, p) * s + k * np.dot(k, p) * (1 - c)
        new_p = new_p / np.linalg.norm(new_p)
        
        new_points = points.copy()
        new_points[idx] = new_p
        
        new_cr = _covering_radius(new_points)
        
        if new_cr < current_cr or rng.random() < np.exp(-(new_cr - current_cr) / t):
            points = new_points
            current_cr = new_cr
            if current_cr < best_cr:
                best_cr = current_cr
                best_points = points.copy()
        
        t = max(t * 0.999, t_min)
    
    return best_points, best_cr


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Start with Fibonacci spiral
    points = _fibonacci_points(N)
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    # Phase 1: Local optimization from Fibonacci start
    if time.time() < deadline - 20:
        points, cr = _local_optimize(points, max_iter=3000)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    # Phase 2: Simulated annealing
    if time.time() < deadline - 10:
        points, cr = _simulated_annealing(best_points, max_iter=2000)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    # Phase 3: Final local polish
    if time.time() < deadline - 5:
        points, cr = _local_optimize(best_points, max_iter=5000, lr_init=0.005)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    # Normalize final result
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
