"""
Strategy: Stochastic Combinatorial Search via Simulated Annealing.
Instead of smooth gradient descent, we treat the configuration as a discrete state space.
We initialize with a Fibonacci spiral (a good uniform baseline) and then perform
random point-to-point swaps. For each swap, we calculate the change in the covering radius.
If the swap reduces the covering radius, we accept it. If it increases the radius,
we accept it with a probability exp(-delta_radius / temperature).
This allows the algorithm to escape local minima that trap gradient-based methods,
particularly by "jumping" over energy barriers to find configurations where
the worst-case gap is smaller.

The covering radius is computed exactly via the convex hull facets (circumradii).
We use a time-budgeted loop to maximize the number of evaluations.
"""

import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def _covering_radius(points):
    """
    Compute the covering radius in degrees.
    Returns 180.0 if the points are not in general position or hull fails.
    """
    try:
        hull = ConvexHull(points)
        facets = hull.simplices
        
        # Extract facet vertices
        p1 = points[facets[:, 0]]
        p2 = points[facets[:, 1]]
        p3 = points[facets[:, 2]]
        
        # Compute circumcenter of each spherical triangle (facet)
        # The circumcenter of a spherical triangle is the point on the sphere
        # equidistant from the three vertices.
        # It is in the direction of the cross product of the edge vectors, 
        # or more robustly, the normalized cross product of (p1-p2) and (p1-p3)
        # doesn't work directly for sphere circumcenters in 3D coords without care.
        
        # A robust way: The circumcenter C is orthogonal to (p1-p2) and (p3-p2) ? No.
        # C is in the plane defined by p1, p2, p3? No, C is on the sphere.
        # The vector C is perpendicular to the plane of the triangle? No.
        
        # Let's use the property that C is equidistant from p1, p2, p3.
        # C . p1 = C . p2 = C . p3 = cos(theta)
        # This implies C is in the null space of the matrix formed by differences?
        # Actually, C is parallel to the cross product of the normal of the plane 
        # and the centroid? 
        # Simpler: C is the unit vector in the direction of:
        # n = (p1 - p2) x (p1 - p3) is normal to the plane.
        # The circumcenter is the intersection of the sphere and the line passing 
        # through the origin and... wait.
        
        # Standard formula for circumcenter of a triangle on the unit sphere:
        # It is the normalized vector:
        # c = (p1 + p2 + p3) is NOT the circumcenter.
        # 
        # Let's solve for c such that c . p1 = c . p2 = c . p3.
        # c . (p1 - p2) = 0
        # c . (p1 - p3) = 0
        # c . c = 1
        # So c is orthogonal to the plane spanned by (p1-p2) and (p1-p3).
        # Thus c is parallel to the cross product (p1-p2) x (p1-p3).
        
        v1 = p1 - p2
        v2 = p1 - p3
        c_dir = np.cross(v1, v2)
        norms = np.linalg.norm(c_dir, axis=1)
        
        valid = norms > 1e-10
        if not np.any(valid):
            return 90.0
            
        c = np.zeros_like(c_dir)
        c[valid] = c_dir[valid] / norms[valid, np.newaxis]
        
        # Orient c towards the centroid of the triangle to ensure it's the "inner" circumcenter
        # (the one inside the spherical triangle, not the antipodal one)
        centroids = (p1 + p2 + p3) / 3.0
        c_norms = np.linalg.norm(centroids, axis=1)
        valid_centroids = c_norms > 1e-10
        centroids[valid_centroids] /= c_norms[valid_centroids, np.newaxis]
        
        dots = np.sum(c * centroids, axis=1)
        c[dots < 0] *= -1
        
        # Angular distance from c to p1 is the covering radius for this facet
        cos_theta = np.sum(c * p1, axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta = np.arccos(cos_theta)
        
        theta[~valid] = 0.0
        return np.max(theta) * 180.0 / np.pi
        
    except Exception:
        return 180.0

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    # Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = _covering_radius(points)
    
    # Simulated Annealing Parameters
    T = 0.1  # Initial temperature (in degrees)
    T_min = 0.001
    cooling_rate = 0.9995
    
    current_points = points.copy()
    current_radius = best_radius
    
    # Precompute random indices for swaps to save time? 
    # np.random.randint is fast enough.
    
    while time.time() < deadline:
        # Generate a random swap: pick two distinct indices i, j
        i, j = np.random.randint(0, N, 2)
        if i == j:
            continue
            
        # Perform swap
        old_i = current_points[i].copy()
        old_j = current_points[j].copy()
        current_points[i] = old_j
        current_points[j] = old_i
        
        # Evaluate new radius
        new_radius = _covering_radius(current_points)
        
        # Acceptance criterion
        if new_radius < current_radius:
            # Accept improvement
            current_radius = new_radius
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = current_points.copy()
        else:
            # Deterioration
            delta = new_radius - current_radius
            # Boltzmann factor
            if T > T_min:
                prob = np.exp(-delta / T)
                if np.random.rand() < prob:
                    # Accept deterioration
                    current_radius = new_radius
                else:
                    # Reject, swap back
                    current_points[i] = old_i
                    current_points[j] = old_j
            
            # Cool down
            T *= cooling_rate
            if T < T_min:
                T = T_min
                # Optional: Restart with a higher T if stuck for too long?
                # For now, just keep going with low T (greedy hill climbing with noise)
        
        # Periodic check to ensure we don't miss a global best due to bad swaps
        # (Not strictly necessary if we track best, but good for debugging)
        
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(f"Covering radius: {_covering_radius(pts)} degrees")
    print(pts.shape)
