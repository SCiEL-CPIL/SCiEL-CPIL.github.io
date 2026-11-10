"""
Optimized sphere covering for N=50 points.
Strategy: Voronoi-based local search with exact convex hull evaluation.
"""
import time
import numpy as np
from scipy.spatial import ConvexHull

N = 50
DEADLINE = 145.0  # Safety margin before 150s limit
START_TIME = time.time()

def _time_remaining():
    return DEADLINE - (time.time() - START_TIME)

def _fibonacci_spiral(n):
    """Generate points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _spherical_circumcenter(a, b, c):
    """
    Compute the circumcenter of a spherical triangle with vertices a, b, c (unit vectors).
    Returns the normalized circumcenter direction.
    """
    # The circumcenter of a spherical triangle is proportional to a*sin(A) + b*sin(B) + c*sin(C)
    # where A, B, C are the angles at the vertices.
    # Alternatively, it is the intersection of the great circles perpendicular to the edges.
    # A robust vector formula for the circumcenter of a triangle on the unit sphere is:
    # CC = (a * sin(2A) + b * sin(2B) + c * sin(2C)) is not quite right directly.
    
    # Better approach: The circumcenter is the normalized vector:
    # (a x b) x (b x c) ... no.
    
    # Let's use the property that the circumcenter p satisfies p.a = p.b = p.c.
    # This means p is in the null space of the matrix formed by (a-b) and (a-c) projected?
    # Actually, the circumcenter is the normalized vector:
    # (a * ||b x c||^2 + b * ||c x a||^2 + c * ||a x b||^2) is NOT correct for spherical.
    
    # Correct formula for spherical circumcenter of triangle (a,b,c):
    # It is the normalized vector:
    # (a * sin(2*angle_a) + b * sin(2*angle_b) + c * sin(2*angle_c))
    # where angle_a is the angle at vertex a.
    
    # Let's compute the angles.
    # cos A = (b-a).(c-a) / (||b-a|| ||c-a||)
    # This is computationally expensive for every facet in every iteration.
    
    # Alternative robust method:
    # The circumcenter is the normalized vector:
    # v = (a x b) x (b x c) + ... no.
    
    # Let's use the linear system approach.
    # p . a = p . b = p . c = h
    # p . (a-b) = 0
    # p . (a-c) = 0
    # So p is orthogonal to (a-b) and (a-c).
    # p is parallel to (a-b) x (a-c).
    # Let's check if this is the circumcenter.
    # If p is parallel to (a-b) x (a-c), then p . a = p . b = p . c.
    # Yes, any vector orthogonal to a-b and a-c has equal dot product with a, b, and c.
    # So the circumcenter direction is simply the normalized cross product of the edge vectors.
    
    edge1 = a - b
    edge2 = a - c
    cross = np.cross(edge1, edge2)
    norm = np.linalg.norm(cross)
    if norm < 1e-12:
        return None
    return cross / norm

def _cover_radius_exact(pts):
    """
    Compute the exact covering radius using convex hull facets.
    Returns (radius_in_radians, worst_facet_indices, circumcenter).
    """
    if len(pts) < 3:
        return np.pi, None, None
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: use dense sampling
        n_samples = 5000
        i = np.arange(n_samples) + 0.5
        z = 1.0 - 2.0 * i / n_samples
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        dots = samples @ pts.T
        min_dots = np.min(dots, axis=1)
        min_dots = np.clip(min_dots, -1.0, 1.0)
        angles = np.arccos(min_dots)
        worst_idx = np.argmax(angles)
        return float(angles[worst_idx]), None, None
    
    max_angle = 0.0
    worst_facet = None
    worst_cc = None
    
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        cc = _spherical_circumcenter(a, b, c)
        if cc is None:
            continue
        
        # The covering radius for this facet is the angle from the circumcenter to the nearest vertex
        # Actually, for a spherical triangle, the circumcenter is equidistant from all three vertices.
        # So the distance to any vertex is the circumradius.
        dot_a = np.dot(cc, a)
        dot_a = np.clip(dot_a, -1.0, 1.0)
        angle = np.arccos(dot_a)
        
        # Check if the circumcenter is "inside" the triangle (empty cap condition)
        # For a convex hull facet, the circumcenter might be outside if the triangle is obtuse.
        # However, for the covering problem, we care about the largest empty cap.
        # If the circumcenter is outside the triangle, the worst point in that cap is one of the vertices or on the edge.
        # But the convex hull facets form a partition of the sphere surface (Voronoi cells are not necessarily triangles if points are not in general position, but convex hull facets are triangles).
        # Wait, the Voronoi cell of a point is the set of points closer to it than to any other.
        # The covering radius is the max over all y of min_i angle(y, p_i).
        # This is equivalent to the max circumradius of the Voronoi cells.
        # The Voronoi cells on the sphere are spherical polygons.
        # The convex hull facets correspond to the dual of the Voronoi diagram? No.
        # The convex hull of the points on the sphere consists of triangles.
        # The circumcenter of a convex hull triangle is NOT necessarily the center of the Voronoi cell.
        
        # Correction:
        # The covering radius is determined by the vertices of the Voronoi diagram.
        # The vertices of the Voronoi diagram on the sphere are the circumcenters of the triangles in the Delaunay triangulation.
        # For points on a sphere, the Delaunay triangulation is the projection of the convex hull facets?
        # Yes, the convex hull of points on the unit sphere gives the Delaunay triangulation.
        # So the circumcenter of each convex hull facet is a vertex of the Voronoi diagram.
        # The covering radius is the maximum distance from any such circumcenter to its nearest point.
        # Since the circumcenter is equidistant from the 3 vertices of the facet, the distance to the nearest point is the distance to any of those 3 vertices, PROVIDED that the circumcenter is inside the facet (i.e., the triangle is acute or the circumcenter falls within the spherical triangle).
        # If the triangle is obtuse, the circumcenter lies outside the triangle. In that case, the Voronoi vertex is not the circumcenter of that specific triangle?
        # Actually, if the triangle is obtuse, the circumcenter is outside, and the Voronoi vertex is determined by the other points.
        # However, for a Delaunay triangulation, the circumcircles (caps) are empty.
        # So the circumcenter is the center of an empty cap.
        # The radius of this cap is the distance to the nearest point, which is one of the 3 vertices.
        # So the calculation `angle = arccos(dot(cc, a))` is correct for the radius of the empty cap centered at cc.
        # We just need to make sure we consider all such caps.
        
        if angle > max_angle:
            max_angle = angle
            worst_facet = simplex
            worst_cc = cc
    
    return max_angle, worst_facet, worst_cc

def _cover_radius_sample(pts, n_samples=10000):
    """Fast covering radius estimate via dense sampling."""
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    dots = samples @ pts.T
    min_dots = np.min(dots, axis=1)
    min_dots = np.clip(min_dots, -1.0, 1.0)
    angles = np.arccos(min_dots)
    return float(np.max(angles))

def _local_search(pts, time_budget=20.0):
    """
    Local search using Voronoi facet information.
    Moves vertices of the worst facet towards its circumcenter.
    """
    pts = pts.copy()
    start_time = time.time()
    best_radius = _cover_radius_exact(pts)[0]
    step = 0.01
    no_improve = 0
    
    while time.time() - start_time < time_budget and _time_remaining() > 5:
        radius, worst_facet, worst_cc = _cover_radius_exact(pts)
        
        if worst_facet is None:
            break
            
        if radius < best_radius:
            best_radius = radius
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 50:
                break
            step *= 0.95
            if step < 1e-6:
                break
        
        # Move vertices of the worst facet towards the circumcenter
        new_pts = pts.copy()
        for idx in worst_facet:
            # Direction from vertex to circumcenter
            direction = worst_cc - pts[idx]
            dir_norm = np.linalg.norm(direction)
            if dir_norm > 1e-10:
                direction /= dir_norm
                new_pts[idx] = pts[idx] + step * direction
        
        new_pts = _normalize(new_pts)
        
        # Check if the move improves the radius
        new_radius = _cover_radius_exact(new_pts)[0]
        if new_radius < radius - 1e-12:
            pts = new_pts
        else:
            # Try a smaller step or different move
            step *= 0.8
            
    return pts, best_radius

def _anneal(pts, time_budget=30.0):
    """
    Simulated annealing to escape local minima.
    Uses high-resolution sampling to find the global worst spot and moves a random point towards it.
    """
    pts = pts.copy()
    start_time = time.time()
    best_radius = _cover_radius_sample(pts, 5000)
    best_pts = pts.copy()
    T = 0.05
    T_min = 0.001
    
    while time.time() - start_time < time_budget and _time_remaining() > 5:
        # Find the worst spot using sampling
        n_samples = 5000
        i = np.arange(n_samples) + 0.5
        z = 1.0 - 2.0 * i / n_samples
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        dots = samples @ pts.T
        min_dots = np.min(dots, axis=1)
        min_dots = np.clip(min_dots, -1.0, 1.0)
        angles = np.arccos(min_dots)
        worst_idx = np.argmax(angles)
        worst_sample = samples[worst_idx]
        current_radius = angles[worst_idx]
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_pts = pts.copy()
            
        # Move a random point towards the worst spot
        idx = np.random.randint(N)
        new_pts = pts.copy()
        direction = worst_sample - pts[idx]
        dir_norm = np.linalg.norm(direction)
        if dir_norm > 1e-10:
            direction /= dir_norm
            new_pts[idx] = pts[idx] + T * direction
        new_pts = _normalize(new_pts)
        
        new_radius = _cover_radius_sample(new_pts, 5000)
        
        # Accept if better or with probability exp(-delta/T)
        if new_radius < current_radius or np.random.random() < np.exp(-(new_radius - current_radius) / T):
            pts = new_pts
        
        T *= 0.99
        if T < T_min:
            break
            
    return best_pts, best_radius

def construct():
    global START_TIME
    START_TIME = time.time()
    
    # Initial configuration: Fibonacci spiral
    pts = _fibonacci_spiral(N)
    best_pts = pts.copy()
    best_radius = _cover_radius_exact(best_pts)[0]
    
    # Phase 1: Local search from Fibonacci start
    if _time_remaining() > 30:
        optimized, r1 = _local_search(pts, time_budget=20.0)
        if r1 < best_radius:
            best_pts = optimized
            best_radius = r1
    
    # Phase 2: Annealing to escape local minima
    if _time_remaining() > 30:
        optimized, r2 = _anneal(best_pts, time_budget=30.0)
        if r2 < best_radius:
            best_pts = optimized
            best_radius = r2
    
    # Phase 3: Local search again after annealing
    if _time_remaining() > 20:
        optimized, r3 = _local_search(best_pts, time_budget=15.0)
        if r3 < best_radius:
            best_pts = optimized
            best_radius = r3
    
    # Phase 4: Multi-start with random initializations
    rng = np.random.RandomState(42)
    while _time_remaining() > 15:
        # Random initialization
        random_pts = rng.randn(N, 3)
        random_pts = _normalize(random_pts)
        
        # Quick local search
        optimized, r4 = _local_search(random_pts, time_budget=5.0)
        if r4 < best_radius:
            best_pts = optimized
            best_radius = r4
            
    # Final check: ensure points are valid
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_spiral(N)
        best_pts = _normalize(best_pts)
    
    # Normalize to ensure unit sphere
    best_pts = _normalize(best_pts)
    
    # Validate with convex hull to ensure no degenerate configurations
    try:
        hull = ConvexHull(best_pts)
        if hull.volume < 1e-6:
            best_pts = _fibonacci_spiral(N)
            best_pts = _normalize(best_pts)
    except Exception:
        best_pts = _fibonacci_spiral(N)
        best_pts = _normalize(best_pts)
    
    return best_pts

