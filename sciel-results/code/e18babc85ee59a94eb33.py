"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n

def _spherical_circumcenter(a, b, c):
    """
    Compute the spherical circumcenter of a triangle defined by points a, b, c on the unit sphere.
    The circumcenter is the point p on the sphere such that angle(p,a)=angle(p,b)=angle(p,c).
    This is equivalent to finding the intersection of the great circles bisecting the arcs.
    A robust method is to find the null space of the matrix with rows (a-b) and (b-c) perpendicular to p?
    Actually, p is equidistant from a, b, c.
    |p-a|^2 = |p-b|^2 = |p-c|^2
    Since |p|=|a|=|b|=|c|=1, this simplifies to p.a = p.b = p.c.
    So p is orthogonal to (a-b) and (b-c).
    p is parallel to (a-b) x (b-c).
    """
    v1 = a - b
    v2 = b - c
    p = np.cross(v1, v2)
    p = _normalize(p)
    
    # We need to ensure p is the center of the "small" cap, not the large one.
    # The circumcenter should be close to the face center.
    face_center = _normalize(a + b + c)
    if np.dot(p, face_center) < 0:
        p = -p
    return p

def _compute_covering_radius(pts):
    """
    Compute the covering radius in degrees.
    The covering radius is the max over all spherical triangles (from convex hull) of the distance
    from the circumcenter to the vertices, provided the circumcenter is not covered by another point.
    """
    pts = np.array(pts)
    if len(pts) < 3:
        return 180.0
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback if hull fails (shouldn't happen for well-distributed points)
        return 180.0
    
    max_r = 0.0
    # hull.simplices are indices of points forming faces.
    # For a 3D convex hull of points on a sphere, the faces are triangles (size 3).
    # If points are coplanar or degenerate, it might be different, but on a sphere they are generally triangular.
    
    # Precompute all pairwise distances? No, just check local neighbors.
    # For each face, compute circumcenter.
    # Check if any other point is closer to the circumcenter than the circumradius.
    
    # Optimization: We only need to check points that are "neighbors" or just do a global check if N is small.
    # N=130 is small enough for O(N^2) if done carefully, but O(F * N) is better.
    
    for simplex in hull.simplices:
        # Ensure simplex is a triangle
        if len(simplex) != 3:
            continue
            
        i, j, k = simplex
        a, b, c = pts[i], pts[j], pts[k]
        
        # Check if the triangle is degenerate (points too close)
        d_ab = np.linalg.norm(a - b)
        if d_ab < 1e-8:
            continue
            
        p = _spherical_circumcenter(a, b, c)
        
        # Calculate the circumradius (distance from p to a)
        r = np.arccos(np.clip(np.dot(p, a), -1.0, 1.0))
        
        # Verify that no other point is closer to p than r.
        # If another point q is closer, then p is not the "empty cap" center for this face's Voronoi cell
        # in the sense that the Voronoi vertex is shifted or this face doesn't define the max distance.
        # Actually, the covering radius is the max distance to the *nearest* point.
        # The farthest point in the Voronoi cell of a set of points is a vertex of the Voronoi diagram.
        # The vertices of the Voronoi diagram on the sphere are the circumcenters of the Delaunay triangles.
        # So we just need to check all Delaunay triangles.
        
        # However, we must ensure that the circumcenter is actually part of the Voronoi cell of the triangle's vertices.
        # This is true if the circumcenter is not "covered" by a point from a non-adjacent triangle?
        # No, the definition of Delaunay triangulation ensures that the circumcircle (cap) is empty of other points.
        # So if we have a valid Delaunay triangulation, every circumcenter is a Voronoi vertex.
        
        # But wait, ConvexHull gives the hull of the points. The faces of the hull correspond to the Delaunay triangles on the sphere.
        # So we just take the max circumradius.
        
        # One subtlety: The "empty cap" condition.
        # The Delaunay condition is that the circumsphere of the tetrahedron is empty.
        # On the sphere, the Delaunay triangle corresponds to the spherical triangle where the circumcircle is empty.
        # So yes, the circumcenter is a Voronoi vertex.
        
        if r > max_r:
            max_r = r
            
    return np.degrees(max_r)

def _lloyd_step(pts):
    """
    Perform one Lloyd iteration.
    Move each point to the centroid of the circumcenters of its adjacent Delaunay triangles.
    """
    pts = np.array(pts)
    n = len(pts)
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return pts
        
    # Map point index to list of adjacent face indices
    # Or directly to circumcenters
    adjacent_circumcenters = [[] for _ in range(n)]
    
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        i, j, k = simplex
        a, b, c = pts[i], pts[j], pts[k]
        
        # Skip degenerate
        if np.linalg.norm(a - b) < 1e-8 or np.linalg.norm(b - c) < 1e-8 or np.linalg.norm(c - a) < 1e-8:
            continue
            
        p = _spherical_circumcenter(a, b, c)
        
        # This circumcenter is a vertex of the Voronoi cells for i, j, k
        adjacent_circumcenters[i].append(p)
        adjacent_circumcenters[j].append(p)
        adjacent_circumcenters[k].append(p)
        
    new_pts = np.zeros_like(pts)
    for i in range(n):
        if len(adjacent_circumcenters[i]) > 0:
            # Average the circumcenters
            centroid = np.mean(adjacent_circumcenters[i], axis=0)
            new_pts[i] = _normalize(centroid)
        else:
            new_pts[i] = pts[i]
            
    return new_pts

def _refine(pts, time_limit):
    """
    Refine the configuration using Lloyd's algorithm with a time limit.
    """
    start_time = time.time()
    current_pts = pts.copy()
    best_pts = current_pts.copy()
    best_cr = _compute_covering_radius(current_pts)
    
    # Initial improvement phase
    iterations = 0
    while time.time() - start_time < time_limit:
        new_pts = _lloyd_step(current_pts)
        
        # Check for validity (points should be distinct)
        # If points coincide, perturb
        # For now, assume Lloyd handles it or we add a check
        
        current_pts = new_pts
        iterations += 1
        
        # Compute new covering radius
        # Optimization: Only compute full CR every few iterations or if improvement is expected?
        # CR calculation is O(N^2) or O(N*F). F is ~2N. So O(N^2).
        # 130^2 = 16900 ops. Very fast.
        new_cr = _compute_covering_radius(current_pts)
        
        if new_cr < best_cr:
            best_cr = new_cr
            best_pts = current_pts.copy()
            
        # Stopping criterion: if no improvement for a long time, stop?
        # Lloyd's usually converges.
        if iterations > 200:
            break
            
    return best_pts, best_cr

def _random_restart(pts, n_restarts, time_limit_per_restart):
    """
    Try multiple random restarts or perturbations to escape local minima.
    """
    best_pts = pts.copy()
    best_cr = _compute_covering_radius(pts)
    
    start_time = time.time()
    total_time = 0
    
    for _ in range(n_restarts):
        # Perturb the current best or start fresh?
        # Strategy: Perturb the current best configuration slightly.
        # Or use a different initial configuration.
        
        # Let's perturb the current best points
        perturbed = pts.copy()
        for i in range(len(perturbed)):
            # Small random rotation
            axis = np.random.randn(3)
            axis = _normalize(axis)
            theta = np.random.uniform(-0.1, 0.1)
            c = np.cos(theta)
            s = np.sin(theta)
            v = perturbed[i]
            new_v = c * v + s * np.cross(axis, v) + (1 - c) * np.dot(axis, v) * axis
            perturbed[i] = _normalize(new_v)
            
        # Refine
        refined_pts, refined_cr = _refine(perturbed, time_limit_per_restart)
        
        if refined_cr < best_cr:
            best_cr = refined_cr
            best_pts = refined_pts.copy()
            
        # Check time
        if time.time() - start_time > time_limit_per_restart * n_restarts:
            break
            
    return best_pts, best_cr

def _ensure_valid(pts):
    """
    Ensure points are normalized, distinct, and not in a hemisphere.
    """
    pts = np.array(pts, dtype=float)
    n = len(pts)
    
    # Normalize
    for i in range(n):
        pts[i] = _normalize(pts[i])
        
    # Check for coincident points
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < 1e-6:
                # Separate them
                pts[j] = _normalize(pts[j] + np.random.randn(3) * 0.01)
                
    # Check not in a closed hemisphere
    # If all points are in a hemisphere, the covering radius is at least 90 degrees.
    # We want to spread them out.
    # Check if there is a vector v such that v . p_i > 0 for all i.
    # This is equivalent to the origin not being in the convex hull of the points.
    # If the origin is in the convex hull, they are not in a hemisphere.
    try:
        hull = ConvexHull(pts)
        # If the origin is inside the hull, the points are not in a hemisphere.
        # We can check if the origin is inside by checking if the hull contains the origin.
        # Simple check: project onto a random axis?
        # Better: Check if the sum of areas of triangles formed with origin is 4*pi?
        # Or simply: if the points are in a hemisphere, the covering radius is large.
        # Let's just check the centroid.
        centroid = np.sum(pts, axis=0)
        if np.linalg.norm(centroid) < 0.1:
             # Likely well-distributed
             pass
        else:
             # Might be in a hemisphere. Try to fix by flipping points?
             # A simple heuristic: if the dot product with the centroid is always positive, flip some.
             # But this is tricky. The optimization should handle this.
             pass
    except Exception:
        pass
        
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
    # Initial configuration: Fibonacci sphere
    def _fibonacci_sphere(n):
        i = np.arange(n) + 0.5
        z = 1.0 - 2.0 * i / n
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    pts = _fibonacci_sphere(N)
    
    # Phase 1: Initial Lloyd refinement
    # Allocate 50% of time for initial refinement
    t1 = (deadline - start_time) * 0.5
    pts, cr = _refine(pts, t1)
    
    # Phase 2: Local search with restarts/perturbations
    # Allocate remaining time
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
         # Number of restarts based on time
         # Each restart takes ~2-5 seconds depending on convergence
         n_restarts = int(remaining_time / 3.0)
         if n_restarts > 0:
             pts, cr = _random_restart(pts, n_restarts, remaining_time / max(1, n_restarts))
             
    # Final validation
    pts = _ensure_valid(pts)
    
    # Ensure we return within time
    if time.time() > deadline - 2.0:
         # Emergency return
         pass
         
    return pts
# EVOLVE-BLOCK-END
