"""Spherical Lloyd's algorithm with proper Voronoi cell centroids and local optimization."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 50

def _facet_circumcenter_and_radius(a, b, c):
    """Compute the circumcenter and circumradius of a spherical triangle.
    
    Returns (circumcenter, circumradius) where circumradius is in radians.
    """
    # The circumcenter is the point equidistant from all three vertices
    # p·a = p·b = p·c means p is orthogonal to (a-b) and (a-c)
    circum_dir = np.cross(a - b, a - c)
    circum_norm = np.linalg.norm(circum_dir)
    
    if circum_norm < 1e-14:
        # Degenerate: points are nearly collinear on the sphere
        # Use the normalized sum as fallback
        centroid = (a + b + c)
        n = np.linalg.norm(centroid)
        if n < 1e-14:
            return None, 0.0
        cc = centroid / n
        # Circumradius is max angle to any vertex
        angles = [np.arccos(np.clip(np.dot(cc, v), -1, 1)) for v in [a, b, c]]
        return cc, max(angles)
    
    cc = circum_dir / circum_norm
    
    # Ensure circumcenter is on the same side as the facet
    facet_normal = np.cross(b - a, c - a)
    fn = np.linalg.norm(facet_normal)
    if fn > 1e-14:
        if np.dot(cc, facet_normal) < 0:
            cc = -cc
    
    # Circumradius: angular distance from cc to any vertex
    dot_a = np.clip(np.dot(cc, a), -1.0, 1.0)
    circumradius = np.arccos(dot_a)
    
    return cc, circumradius

def _facet_area(a, b, c):
    """Compute the spherical area of a triangle with vertices a, b, c (unit vectors)."""
    # Spherical excess: area = A + B + C - pi
    # Where A, B, C are the angles at vertices a, b, c
    # Angle at a: angle between great circle arcs from a to b and a to c
    
    # Using the formula: tan(A/2) = |a × b| / (1 + a·b) ... actually let's use the vector formula
    # For a spherical triangle, the area can be computed as:
    # 2 * arctan(|a·(b×c)| / (1 + a·b + b·c + c·a))
    
    cross_bc = np.cross(b, c)
    dot_abc = np.dot(a, cross_bc)
    denom = 1.0 + np.dot(a, b) + np.dot(b, c) + np.dot(c, a)
    
    if abs(denom) < 1e-14:
        # Edge case
        return 0.0
    
    area = 2.0 * np.arctan2(abs(dot_abc), denom)
    return area

def _compute_covering_radius(points):
    """Compute the covering radius in degrees for a set of points on the sphere."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return 90.0
    
    max_circumradius = 0.0
    
    for simplex in hull.simplices:
        a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        cc, cr = _facet_circumcenter_and_radius(a, b, c)
        if cc is not None:
            if cr > max_circumradius:
                max_circumradius = cr
    
    return np.degrees(max_circumradius)

def _lloyd_step(points):
    """Perform one Lloyd's algorithm step with proper Voronoi cell centroids."""
    new_points = np.zeros_like(points)
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return points
    
    # Build mapping from point index to incident facets
    point_to_facets = {i: [] for i in range(len(points))}
    for simplex in hull.simplices:
        for v in simplex:
            point_to_facets[v].append(simplex)
    
    for i in range(len(points)):
        incident_facets = point_to_facets[i]
        
        if len(incident_facets) == 0:
            new_points[i] = points[i]
            continue
        
        # Compute the Voronoi cell centroid as the area-weighted average of facet circumcenters
        weighted_sum = np.zeros(3)
        total_weight = 0.0
        
        for simplex in incident_facets:
            a, b, c = points[simplex[0]], points[simplex[1]], points[simplex[2]]
            cc, cr = _facet_circumcenter_and_radius(a, b, c)
            area = _facet_area(a, b, c)
            
            if cc is not None and area > 1e-14:
                weighted_sum += area * cc
                total_weight += area
        
        if total_weight > 1e-14:
            centroid = weighted_sum / total_weight
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm > 1e-14:
                new_points[i] = centroid / centroid_norm
            else:
                new_points[i] = points[i]
        else:
            new_points[i] = points[i]
    
    return new_points

def _local_optimize(points, max_time=5.0):
    """Local optimization: minimize the maximum circumradius by adjusting points slightly."""
    start_time = time.time()
    current_points = points.copy()
    current_radius = _compute_covering_radius(current_points)
    
    # Use gradient-free optimization on the largest facets
    # We'll do a few iterations of moving points toward the circumcenters of their largest facets
    
    for _ in range(30):
        if time.time() - start_time > max_time:
            break
        
        try:
            hull = ConvexHull(current_points)
        except Exception:
            break
        
        # Find the facet with the largest circumradius
        max_cr = 0
        worst_facet = None
        for simplex in hull.simplices:
            a, b, c = current_points[simplex[0]], current_points[simplex[1]], current_points[simplex[2]]
            cc, cr = _facet_circumcenter_and_radius(a, b, c)
            if cc is not None and cr > max_cr:
                max_cr = cr
                worst_facet = simplex
        
        if worst_facet is None:
            break
        
        # Move the three vertices of the worst facet toward the circumcenter
        cc, _ = _facet_circumcenter_and_radius(
            current_points[worst_facet[0]], 
            current_points[worst_facet[1]], 
            current_points[worst_facet[2]]
        )
        
        if cc is None:
            break
        
        # Small step toward circumcenter
        step_size = 0.1
        for idx in worst_facet:
            new_pos = current_points[idx] + step_size * (cc - current_points[idx])
            norm = np.linalg.norm(new_pos)
            if norm > 1e-14:
                current_points[idx] = new_pos / norm
        
        # Normalize all points
        norms = np.linalg.norm(current_points, axis=1, keepdims=True)
        current_points = current_points / norms
        
        # Check for coincident points
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(current_points[i] - current_points[j])
                if d < min_dist:
                    min_dist = d
        if min_dist < 1e-8:
            break
        
        new_radius = _compute_covering_radius(current_points)
        if new_radius < current_radius - 1e-6:
            current_radius = new_radius
        else:
            # Revert if no improvement
            break
    
    return current_points

def construct():
    """Construct 50 points on the unit sphere using improved spherical Lloyd's algorithm."""
    # Start with a good initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize to ensure unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Run Lloyd's algorithm iterations
    start_time = time.time()
    max_time = 100.0  # Leave time for local optimization
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    for iteration in range(300):
        if time.time() - start_time > max_time:
            break
        
        new_points = _lloyd_step(points)
        
        # Normalize
        norms = np.linalg.norm(new_points, axis=1, keepdims=True)
        new_points = new_points / norms
        
        # Check if points are distinct
        min_dist = np.inf
        for i in range(N):
            for j in range(i+1, N):
                d = np.linalg.norm(new_points[i] - new_points[j])
                if d < min_dist:
                    min_dist = d
        if min_dist < 1e-10:
            continue
        
        radius = _compute_covering_radius(new_points)
        
        if radius < best_radius:
            best_radius = radius
            best_points = new_points.copy()
        
        points = new_points
    
    # Local optimization on the best configuration found
    remaining_time = max_time - (time.time() - start_time)
    if remaining_time > 2.0:
        optimized = _local_optimize(best_points, max_time=min(remaining_time - 1.0, 15.0))
        opt_radius = _compute_covering_radius(optimized)
        if opt_radius < best_radius:
            best_points = optimized
            best_radius = opt_radius
    
    return best_points
