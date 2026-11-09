"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

# EVOLVE-BLOCK-START

def _fibonacci_sphere(n):
    """Generate points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return points

def _covering_radius(points):
    """
    Compute the exact covering radius (in radians) using the convex hull.
    The covering radius is the maximum circumradius of the spherical facets.
    Each facet of the convex hull corresponds to a Voronoi cell on the sphere.
    The circumradius of the spherical triangle (or polygon) formed by the facet vertices
    is the distance from the center of that empty cap to the nearest point.
    """
    try:
        hull = ConvexHull(points)
        max_r = 0.0
        for simplex in hull.simplices:
            # simplex contains indices of points forming a triangular face
            v1, v2, v3 = points[simplex]
            
            # The normal to the plane of the triangle
            normal = np.cross(v2 - v1, v3 - v1)
            norm_n = np.linalg.norm(normal)
            if norm_n < 1e-12:
                continue
            normal = normal / norm_n
            
            d = np.dot(normal, v1)
            if d < 0:
                normal = -normal
            
            dot = np.dot(normal, v1)
            dot = np.clip(dot, -1.0, 1.0)
            angle = np.arccos(dot)
            
            if angle > max_r:
                max_r = angle
                
        return max_r
    except Exception:
        return np.pi

def _spherical_centroid(vertices):
    """
    Compute the spherical centroid (mean direction) of a set of unit vectors.
    This is the centroid of the spherical polygon formed by the vertices.
    """
    # Average the vectors
    centroid = np.mean(vertices, axis=0)
    norm = np.linalg.norm(centroid)
    if norm < 1e-12:
        return None
    return centroid / norm

def _lloyd_step(points, alpha=0.5):
    """
    Perform one Lloyd iteration on the sphere.
    For each point, compute the centroid of its Voronoi cell (spherical facet)
    and move the point toward that centroid.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return points.copy()
    
    # Build a mapping from point index to list of facet vertices
    # Each facet is a triangle (simplex) in the convex hull
    # For each point, collect all facets that contain it
    point_to_facets = {i: [] for i in range(N)}
    
    for simplex in hull.simplices:
        for idx in simplex:
            point_to_facets[idx].append(simplex)
    
    new_points = points.copy()
    for i in range(N):
        facets = point_to_facets[i]
        if not facets:
            continue
        
        # Collect all unique vertices from facets containing point i
        # The Voronoi cell of point i is the spherical polygon whose vertices
        # are the centroids of the facets (triangles) in the convex hull
        # that have point i as a vertex.
        
        # Actually, for Lloyd's algorithm on the sphere:
        # The Voronoi cell of point i consists of all points on the sphere
        # closer to i than to any other point.
        # The vertices of this cell are the circumcenters of the spherical triangles
        # (facets of the convex hull) that have i as a vertex.
        
        # So we need to find all facets that contain point i, compute their
        # circumcenters (which are the outward normals of the facet planes),
        # and then compute the spherical centroid of those circumcenters.
        
        circumcenters = []
        for simplex in facets:
            v1, v2, v3 = points[simplex]
            normal = np.cross(v2 - v1, v3 - v1)
            norm_n = np.linalg.norm(normal)
            if norm_n < 1e-12:
                continue
            normal = normal / norm_n
            d = np.dot(normal, v1)
            if d < 0:
                normal = -normal
            circumcenters.append(normal)
        
        if not circumcenters:
            continue
        
        circumcenters = np.array(circumcenters)
        centroid = _spherical_centroid(circumcenters)
        if centroid is not None:
            # Move point i toward the centroid
            new_points[i] = points[i] + alpha * (centroid - points[i])
            norm = np.linalg.norm(new_points[i])
            if norm > 1e-12:
                new_points[i] = new_points[i] / norm
    
    return new_points

def _simulated_annealing_step(points, temp, rng):
    """
    Perform one simulated annealing step.
    Randomly perturb a subset of points and accept if covering radius improves
    or with probability exp(-delta/temp).
    """
    new_points = points.copy()
    
    # Randomly select k points to perturb
    k = rng.integers(1, max(2, N // 5))
    indices = rng.choice(N, size=k, replace=False)
    
    # Perturb these points
    for idx in indices:
        # Random direction perturbation
        perturbation = rng.standard_normal(3)
        perturbation = perturbation / np.linalg.norm(perturbation)
        new_points[idx] = points[idx] + 0.1 * temp * perturbation
        norm = np.linalg.norm(new_points[idx])
        if norm > 1e-12:
            new_points[idx] = new_points[idx] / norm
    
    return new_points

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    # Start with Fibonacci spiral
    points = _fibonacci_sphere(N)
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    # Phase 1: Lloyd's algorithm to get a good local optimum
    lloyd_alpha = 0.5
    for step in range(200):
        if time.time() - start_time > time_limit:
            break
        points = _lloyd_step(points, lloyd_alpha)
        if step % 20 == 0:
            cr = _covering_radius(points)
            if cr < best_cr:
                best_cr = cr
                best_points = points.copy()
        # Decay alpha
        lloyd_alpha *= 0.99
    
    # Phase 2: Simulated Annealing for global exploration
    # Start temperature based on current covering radius
    temp = best_cr * 0.5
    min_temp = 1e-4
    cooling_rate = 0.995
    
    sa_points = best_points.copy()
    sa_cr = best_cr
    
    while time.time() - start_time < time_limit:
        new_points = _simulated_annealing_step(sa_points, temp, rng)
        
        # Evaluate new configuration
        new_cr = _covering_radius(new_points)
        
        # Accept if better, or with probability exp(-(new_cr - sa_cr)/temp)
        if new_cr < sa_cr:
            sa_points = new_points
            sa_cr = new_cr
            if sa_cr < best_cr:
                best_cr = sa_cr
                best_points = sa_points.copy()
        else:
            delta = new_cr - sa_cr
            if delta < 1e-10:
                sa_points = new_points
                sa_cr = new_cr
            else:
                accept_prob = np.exp(-delta / max(temp, 1e-10))
                if rng.random() < accept_prob:
                    sa_points = new_points
                    sa_cr = new_cr
        
        # Cool down
        temp = max(temp * cooling_rate, min_temp)
        
        # Periodically re-run Lloyd from current SA state
        if int((time.time() - start_time) * 10) % 50 == 0 and time.time() - start_time < time_limit - 10:
            lloyd_points = sa_points.copy()
            for lstep in range(30):
                if time.time() - start_time > time_limit:
                    break
                lloyd_points = _lloyd_step(lloyd_points, 0.3)
            lloyd_cr = _covering_radius(lloyd_points)
            if lloyd_cr < best_cr:
                best_cr = lloyd_cr
                best_points = lloyd_points.copy()
            # Update SA state with improved Lloyd result
            if lloyd_cr < sa_cr:
                sa_points = lloyd_points.copy()
                sa_cr = lloyd_cr
    
    # Final Lloyd refinement
    if time.time() - start_time < time_limit - 5:
        final_points = best_points.copy()
        for fstep in range(50):
            if time.time() - start_time > time_limit:
                break
            final_points = _lloyd_step(final_points, 0.2)
        final_cr = _covering_radius(final_points)
        if final_cr < best_cr:
            best_cr = final_cr
            best_points = final_points.copy()
    
    return best_points

# EVOLVE-BLOCK-END
