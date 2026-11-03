"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 50

# EVOLVE-BLOCK-START

def _compute_covering_radius(points):
    """
    Compute the covering radius for a set of points on the unit sphere.
    The covering radius is the maximum angular distance from any point on the sphere
    to the nearest of the given points.
    This is equivalent to the maximum circumradius of the spherical triangles
    in the Delaunay triangulation (convex hull facets).
    """
    # Use scipy to compute the convex hull
    from scipy.spatial import ConvexHull
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # If hull computation fails (e.g., degenerate points), return a large value
        return np.pi
        
    max_radius = 0.0
    
    # Iterate over the facets of the convex hull
    for simplex in hull.simplices:
        a = points[simplex[0]]
        b = points[simplex[1]]
        c = points[simplex[2]]
        
        # Compute the normal to the plane containing a, b, c
        # This vector is perpendicular to (b-a) and (c-a)
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        
        if n_norm < 1e-10:
            # Degenerate triangle
            continue
            
        u = n / n_norm
        
        # Determine the correct sign for the circumcenter
        # The circumcenter should be on the same side of the triangle as the centroid
        centroid = (a + b + c) / 3.0
        if np.dot(u, centroid) < 0:
            u = -u
            
        # The angular distance from u to any of a, b, c is the circumradius
        # Since u is on the sphere and a is on the sphere, the distance is arccos(u dot a)
        cos_r = np.dot(u, a)
        # Clip to [-1, 1] to avoid numerical errors in arccos
        cos_r = np.clip(cos_r, -1.0, 1.0)
        r = np.arccos(cos_r)
        
        if r > max_radius:
            max_radius = r
            
    return max_radius

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin
    
    # 1. Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Compute initial covering radius
    current_radius = _compute_covering_radius(points)
    best_points = points.copy()
    best_radius = current_radius
    
    # 3. Simulated Annealing
    T = 0.01  # Initial temperature
    T_min = 1e-6
    cooling_rate = 0.99
    
    while time.time() < deadline:
        if T < T_min:
            break
            
        # Pick a random point to perturb
        idx = np.random.randint(N)
        original_point = points[idx].copy()
        
        # Perturb the point
        # Add random noise and renormalize
        noise = np.random.randn(3) * 0.1
        new_point = original_point + noise
        new_point /= np.linalg.norm(new_point)
        
        points[idx] = new_point
        
        # Compute new radius
        new_radius = _compute_covering_radius(points)
        
        # Acceptance criterion
        delta = new_radius - current_radius
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_radius = new_radius
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = points.copy()
        else:
            # Reject
            points[idx] = original_point
            
        # Cool down
        T *= cooling_rate
        
    return best_points

# EVOLVE-BLOCK-END
