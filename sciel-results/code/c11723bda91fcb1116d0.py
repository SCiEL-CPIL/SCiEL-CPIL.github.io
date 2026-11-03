"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay

N = 50

# EVOLVE-BLOCK-START
def _normalize(v):
    """Normalize rows of v to unit length."""
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return v / norms


def _covering_radius(points):
    """
    Compute the covering radius (in degrees) of points on the unit sphere.
    Uses the convex hull: the covering radius is the maximum circumradius
    of the spherical Voronoi cells, which equals the maximum angular distance
    from a point to the farthest vertex of its cell.
    
    For points on the unit sphere, the covering radius is the maximum, over all
    facets of the convex hull, of the circumradius of that facet (angular distance
    from the facet center to its vertices).
    """
    # Compute convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        return 90.0  # fallback if hull fails
    
    # For each facet, compute the circumradius
    # The circumradius of a spherical triangle with vertices on the unit sphere
    # can be computed from the side lengths (angular distances)
    
    # Alternative: for each facet, find the circumcenter and compute max distance
    # But for a convex hull in 3D, each facet is a triangle (or polygon)
    
    # The covering radius is the max over all points on the sphere of the distance
    # to the nearest configuration point. This equals the max circumradius of the
    # Voronoi cells.
    
    # A simpler approach: the covering radius equals the maximum, over all facets
    # of the convex hull, of the angular circumradius of that facet.
    
    # For a triangle with vertices a, b, c on the unit sphere, the circumradius
    # (angular) can be computed as follows:
    # The circumcenter of the triangle on the sphere is the point equidistant from
    # all three vertices. The angular circumradius R satisfies:
    # cos(R) = (cos(ab) + cos(bc) + cos(ca) - 1) / (2 * sqrt((1+cos(ab))(1+cos(bc))(1+cos(ca))))
    # Wait, that's not quite right. Let me use a different approach.
    
    # Actually, for the covering radius, we can use the fact that it's the maximum
    # over all facets of the circumradius. For a planar triangle with side lengths
    # a, b, c (chord lengths), the circumradius is abc/(4*Area).
    # The angular circumradius is then arcsin(R_chord) or we can compute it differently.
    
    # Let me use a more direct approach: for each facet (triangle), compute the
    # point on the sphere that is equidistant from all three vertices (the circumcenter
    # of the spherical triangle), and the distance from that point to each vertex is
    # the circumradius.
    
    # Simpler: use the formula for circumradius of a spherical triangle.
    # Given three points on the unit sphere, the circumradius R (angular) satisfies:
    # sin(R) = (a * b * c) / (4 * sin(s)) where a,b,c are side lengths and s is area?
    # No, let me just compute it numerically.
    
    # For each triangular facet, compute the circumcenter:
    # The circumcenter is the intersection of the sphere with the line perpendicular
    # to the plane of the triangle passing through the centroid... no.
    
    # Let me use a different approach. The covering radius is the maximum over all
    # points y on the sphere of min_i angle(y, p_i). This is equal to the maximum
    # circumradius of the Voronoi cells.
    
    # For the convex hull approach: the Voronoi cells on the sphere correspond to
    # the facets of the convex hull. The circumradius of a Voronoi cell (which is a
    # spherical polygon) is the distance from the cell center to the farthest vertex.
    
    # Actually, let me just sample points on the sphere and compute the covering radius
    # directly. This is slower but more reliable.
    
    # Better approach: use the convex hull facets. For each facet (triangle), the
    # circumradius of the corresponding Voronoi cell can be computed.
    
    # Let me use the approach: for each facet, compute the circumcenter of the
    # spherical triangle and the max distance to vertices.
    
    max_cr = 0.0
    for simplex in hull.simplices:
        tri = points[simplex]
        a, b, c = tri[0], tri[1], tri[2]
        
        # Compute side lengths (angular distances)
        def ang_dist(p, q):
            d = np.dot(p, q)
            d = np.clip(d, -1.0, 1.0)
            return np.arccos(d)
        
        ab = ang_dist(a, b)
        bc = ang_dist(b, c)
        ca = ang_dist(c, a)
        
        # Compute circumradius using spherical law of sines or direct formula
        # For a spherical triangle, the circumradius R satisfies:
        # sin(R) = sin(A) / (2 * sin(a)) where A is angle at vertex A, a is opposite side
        # But we need all angles.
        
        # Alternative: use the formula
        # cos(R) = (cos(ab) * cos(bc) * cos(ca) + 2 * cos(ab) * cos(bc) * cos(ca) - cos(ab)^2 - cos(bc)^2 - cos(ca)^2) / ...
        # This is getting complicated. Let me use a numerical approach.
        
        # The circumcenter of a spherical triangle is the point equidistant from all three.
        # It lies on the perpendicular bisector planes.
        
        # For three points a, b, c on the unit sphere, the circumcenter x satisfies:
        # |x - a|^2 = |x - b|^2 = |x - c|^2
        # which gives (x-a).(x-a) = (x-b).(x-b), etc.
        # x.x - 2x.a + a.a = x.x - 2x.b + b.b
        # -2x.a + 1 = -2x.b + 1 (since a.a = b.b = 1)
        # x.a = x.b
        # Similarly x.a = x.c
        # So x is equidistant (in dot product) from a, b, c.
        # This means x is in the null space of [a-b; a-c] (as a linear system for the dot products).
        
        # The set of points x with x.a = x.b = x.c is a line (intersection of two planes through origin).
        # The circumcenter is the intersection of this line with the unit sphere.
        
        # Solve: x.a = x.b => x.(a-b) = 0
        # x.a = x.c => x.(a-c) = 0
        # So x is perpendicular to (a-b) and (a-c), i.e., x is parallel to (a-b) x (a-c).
        
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        
        if np.linalg.norm(cross) < 1e-10:
            # Degenerate, use centroid
            center = (a + b + c) / 3.0
            center = center / np.linalg.norm(center)
        else:
            center = cross / np.linalg.norm(cross)
        
        # The circumradius is the angular distance from center to any vertex
        d = np.dot(center, a)
        d = np.clip(d, -1.0, 1.0)
        cr = np.arccos(d)
        
        if cr > max_cr:
            max_cr = cr
    
    return np.degrees(max_cr)


def _fibonacci_points(n):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _spherical_centroid(points, weights=None):
    """Compute the spherical centroid (mean direction) of points."""
    if weights is None:
        weights = np.ones(len(points))
    vec = np.sum(points * weights[:, None], axis=0)
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        return points[0].copy()
    return vec / norm


def _lloyd_iteration(points, max_iter=100, tol=1e-10, time_limit=None, start_time=None):
    """
    Spherical Lloyd's algorithm: move each point to the centroid of its
    Voronoi cell (computed via convex hull).
    """
    if time_limit is None:
        time_limit = 100.0
    if start_time is None:
        start_time = time.time()
    
    points = _normalize(points.copy())
    
    for it in range(max_iter):
        # Check time
        if time.time() - start_time > time_limit - 5.0:
            break
        
        # Compute convex hull
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # For each point, find its Voronoi cell center
        # The Voronoi cell of point i corresponds to the facet(s) of the dual polytope
        # In 3D, the Voronoi cell of point i is a polyhedron whose vertices are the
        # circumcenters of the facets containing i.
        
        # Simpler approach: for each point, compute the centroid of the facets
        # that contain it, weighted by the area of each facet.
        
        new_points = points.copy()
        
        # Build mapping from point index to list of facets containing it
        point_to_facets = {i: [] for i in range(len(points))}
        for facet_idx, simplex in enumerate(hull.simplices):
            for v in simplex:
                point_to_facets[v].append(facet_idx)
        
        for i in range(len(points)):
            facets = point_to_facets[i]
            if len(facets) == 0:
                continue
            
            # Compute the spherical centroid of the Voronoi cell
            # Approximate by averaging the circumcenters of adjacent facets
            circumcenters = []
            for fi in facets:
                tri = points[hull.simplices[fi]]
                a, b, c = tri[0], tri[1], tri[2]
                
                v1 = a - b
                v2 = a - c
                cross = np.cross(v1, v2)
                norm_cross = np.linalg.norm(cross)
                if norm_cross > 1e-10:
                    cc = cross / norm_cross
                    circumcenters.append(cc)
            
            if len(circumcenters) == 0:
                continue
            
            # Average the circumcenters
            avg = np.mean(circumcenters, axis=0)
            avg_norm = np.linalg.norm(avg)
            if avg_norm > 1e-10:
                new_points[i] = avg / avg_norm
        
        new_points = _normalize(new_points)
        
        # Check convergence
        diff = np.max(np.linalg.norm(new_points - points, axis=1))
        points = new_points
        
        if diff < tol:
            break
    
    return points


def _simulated_annealing(points, time_limit=20.0, start_time=None):
    """
    Simulated annealing to improve the covering radius.
    """
    if start_time is None:
        start_time = time.time()
    
    points = _normalize(points.copy())
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    T = 5.0  # initial temperature (degrees)
    T_min = 0.01
    cooling = 0.995
    
    while T > T_min:
        if time.time() - start_time > time_limit - 3.0:
            break
        
        # Random move: pick a random point and perturb it
        i = np.random.randint(len(points))
        perturbation = np.random.normal(0, T * np.pi / 180.0, 3)
        new_points = points.copy()
        new_points[i] = new_points[i] + perturbation
        new_points[i] = new_points[i] / np.linalg.norm(new_points[i])
        
        new_cr = _covering_radius(new_points)
        delta = new_cr - best_cr
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            points = new_points
            if new_cr < best_cr:
                best_cr = new_cr
                best_points = points.copy()
        
        T *= cooling
    
    return best_points, best_cr


def _local_optimization(points, time_limit=15.0, start_time=None):
    """
    Local optimization: gradient-based refinement of the covering radius.
    Uses the fact that the covering radius is determined by the largest facet.
    """
    if start_time is None:
        start_time = time.time()
    
    points = _normalize(points.copy())
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    # Find the facet with the largest circumradius
    try:
        hull = ConvexHull(points)
    except Exception:
        return best_points, best_cr
    
    # Compute circumradius for each facet
    max_cr = 0.0
    max_facet = None
    for simplex in hull.simplices:
        tri = points[simplex]
        a, b, c = tri[0], tri[1], tri[2]
        
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        if norm_cross > 1e-10:
            center = cross / norm_cross
            d = np.dot(center, a)
            d = np.clip(d, -1.0, 1.0)
            cr = np.arccos(d)
            if cr > max_cr:
                max_cr = cr
                max_facet = simplex
    
    if max_facet is None:
        return best_points, best_cr
    
    # Move the vertices of the largest facet toward the circumcenter
    # to reduce the circumradius
    tri = points[max_facet]
    a, b, c = tri[0], tri[1], tri[2]
    
    v1 = a - b
    v2 = a - c
    cross = np.cross(v1, v2)
    norm_cross = np.linalg.norm(cross)
    if norm_cross < 1e-10:
        return best_points, best_cr
    
    center = cross / norm_cross
    
    # Move each vertex a small step toward the center
    step_size = 0.01
    new_points = points.copy()
    for idx in max_facet:
        direction = center - points[idx]
        norm_dir = np.linalg.norm(direction)
        if norm_dir > 1e-10:
            new_points[idx] = points[idx] + step_size * direction / norm_dir
            new_points[idx] = new_points[idx] / np.linalg.norm(new_points[idx])
    
    new_cr = _covering_radius(new_points)
    if new_cr < best_cr:
        best_cr = new_cr
        best_points = new_points
    
    return best_points, best_cr


def construct():
    start_time = time.time()
    total_time_limit = 145.0  # safety margin
    
    # Start with Fibonacci spiral
    points = _fibonacci_points(N)
    
    # Phase 1: Lloyd's algorithm
    points = _lloyd_iteration(points, max_iter=50, time_limit=30.0, start_time=start_time)
    cr = _covering_radius(points)
    best_points = points.copy()
    best_cr = cr
    
    # Phase 2: Simulated annealing
    if time.time() - start_time < total_time_limit - 25.0:
        sa_points, sa_cr = _simulated_annealing(points, time_limit=25.0, start_time=start_time)
        if sa_cr < best_cr:
            best_cr = sa_cr
            best_points = sa_points
    
    # Phase 3: Another Lloyd's iteration
    if time.time() - start_time < total_time_limit - 20.0:
        points = _lloyd_iteration(best_points, max_iter=30, time_limit=20.0, start_time=start_time)
        cr = _covering_radius(points)
        if cr < best_cr:
            best_cr = cr
            best_points = points
    
    # Phase 4: Local optimization
    if time.time() - start_time < total_time_limit - 10.0:
        lo_points, lo_cr = _local_optimization(best_points, time_limit=10.0, start_time=start_time)
        if lo_cr < best_cr:
            best_cr = lo_cr
            best_points = lo_points
    
    # Final Lloyd's polish
    if time.time() - start_time < total_time_limit - 5.0:
        points = _lloyd_iteration(best_points, max_iter=20, time_limit=5.0, start_time=start_time)
        cr = _covering_radius(points)
        if cr < best_cr:
            best_cr = cr
            best_points = points
    
    return _normalize(best_points)
# EVOLVE-BLOCK-END
