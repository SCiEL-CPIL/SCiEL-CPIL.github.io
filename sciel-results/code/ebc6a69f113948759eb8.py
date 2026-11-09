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
            
            # The center of the circumcircle on the sphere is in the direction of the normal
            # if the normal points outward. We need to ensure it's on the same side as the origin relative to the plane?
            # Actually, the center of the spherical cap is the point on the sphere in the direction of the outward normal.
            # Let's check if the normal points away from the origin.
            # The plane equation is n . x = d.
            # d = n . v1
            # Since points are on the unit sphere, and the origin is inside the convex hull (assuming good distribution),
            # the distance from origin to plane is d.
            # The center of the cap is normal * (1/|normal|) ? No.
            
            # The point on the sphere farthest from the triangle's plane in the direction of the normal is:
            # c = normal / ||normal||
            # Wait, the normal is already normalized.
            # Is c the center?
            # The distance from c to v1 is the covering radius for this facet.
            # Let's verify: c is a unit vector. v1 is a unit vector.
            # The angle is arccos(c . v1).
            
            # However, we must ensure that the cap centered at c with radius arccos(c.v1) is empty.
            # The convex hull facet is formed by v1, v2, v3.
            # The plane containing v1, v2, v3 does not contain the origin (if points are not in a hemisphere).
            # The normal `normal` points to one side.
            # We want the normal pointing away from the origin so that the cap is on the outside.
            # The vector from origin to plane is d * normal.
            # If d > 0, the origin is on the negative side, so normal points away.
            # If d < 0, normal points towards origin, so we should flip it.
            
            d = np.dot(normal, v1)
            if d < 0:
                normal = -normal
            
            # The center of the spherical cap is `normal` (which is a unit vector).
            # The radius is the angle between `normal` and any of the vertices, e.g., v1.
            # dot = normal . v1
            # Since normal is unit and v1 is unit, dot = cos(angle).
            # angle = arccos(dot)
            
            dot = np.dot(normal, v1)
            # Clip for numerical stability
            dot = np.clip(dot, -1.0, 1.0)
            angle = np.arccos(dot)
            
            if angle > max_r:
                max_r = angle
                
        return max_r
    except Exception:
        return np.pi # Fallback if hull fails (e.g., coplanar points)

def _repulsive_force_step(points, p=20, eta=0.01):
    """
    Perform one step of repulsive force optimization.
    Maximizes the minimum distance between points.
    """
    # Compute pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Avoid division by zero
    mask = ~np.eye(N, dtype=bool)
    dists[mask] = np.maximum(dists[mask], 1e-10)
    
    # Weights for repulsion
    weights = p * (dists ** (-p - 2))
    np.fill_diagonal(weights, 0.0)
    
    # Forces
    sum_weights = np.sum(weights, axis=1)
    weighted_points = np.sum(weights[:, :, np.newaxis] * points[np.newaxis, :, :], axis=1)
    forces = points * sum_weights[:, np.newaxis] - weighted_points
    
    # Update
    points = points + eta * forces
    
    # Normalize
    norms = np.linalg.norm(points, axis=1)
    points = points / norms[:, np.newaxis]
    
    return points

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin
    
    # Start with Fibonacci spiral
    points = _fibonacci_sphere(N)
    
    # Initial covering radius
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    # Optimization parameters
    p = 20
    eta = 0.05
    min_eta = 1e-5
    decay = 0.999
    
    # Precompute mask for non-diagonal elements
    mask = ~np.eye(N, dtype=bool)
    
    step = 0
    while time.time() - start_time < time_limit:
        # Update points
        points = _repulsive_force_step(points, p, eta)
        
        # Evaluate covering radius periodically to save time
        # Every 100 steps or if eta is very small
        if step % 100 == 0 or eta < min_eta * 10:
            cr = _covering_radius(points)
            if cr < best_cr:
                best_cr = cr
                best_points = points.copy()
        
        # Decay step size
        eta *= decay
        
        step += 1
        
        # Check time limit
        if time.time() - start_time > time_limit:
            break
            
    return best_points

# EVOLVE-BLOCK-END
