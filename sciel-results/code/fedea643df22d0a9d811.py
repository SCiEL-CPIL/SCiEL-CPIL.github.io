import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def compute_covering_radius(points):
    """
    Computes the exact covering radius using ConvexHull.
    Returns radius in radians.
    """
    try:
        hull = ConvexHull(points)
        simplices = hull.simplices
        a = points[simplices[:, 0]]
        b = points[simplices[:, 1]]
        c = points[simplices[:, 2]]
        
        # Compute face normals
        n = np.cross(b - a, c - a)
        norms = np.linalg.norm(n, axis=1)
        valid = norms > 1e-10
        if not np.all(valid):
            return np.pi
            
        u = n / norms[:, np.newaxis]
        
        # Ensure normals point outward
        centroids = (a + b + c) / 3.0
        dots = np.sum(u * centroids, axis=1)
        u[dots < 0] *= -1
        
        # Compute circumradius for each facet
        # The circumradius R of a spherical triangle with side lengths a,b,c is tricky.
        # But the covering radius is the max distance from any point on the sphere to the nearest vertex.
        # This is the max over all facets of the distance from the circumcenter to the vertices.
        # For a facet, the "worst" point is the circumcenter.
        # We approximate the circumcenter as the normalized sum of vertices if the triangle is small,
        # or solve for it exactly.
        
        F = len(simplices)
        max_radius = 0.0
        
        for i in range(F):
            ai = a[i]
            bi = b[i]
            ci = c[i]
            
            # Solve for circumcenter cc on the unit sphere equidistant from ai, bi, ci
            # |cc - ai|^2 = |cc - bi|^2 => 2 cc . (bi - ai) = |bi|^2 - |ai|^2 = 0 (since on unit sphere)
            # So cc is orthogonal to (bi-ai) and (ci-ai).
            # Thus cc is parallel to the normal u[i].
            # Let cc = s * u[i]. Since |cc|=1, cc = u[i] or -u[i].
            # We pick the one closer to the centroid (outward normal).
            
            cos_a = np.dot(u[i], ai)
            cos_b = np.dot(u[i], bi)
            cos_c = np.dot(u[i], ci)
            
            d_a = np.arccos(np.clip(cos_a, -1.0, 1.0))
            d_b = np.arccos(np.clip(cos_b, -1.0, 1.0))
            d_c = np.arccos(np.clip(cos_c, -1.0, 1.0))
            
            r1 = np.max([d_a, d_b, d_c])
            
            # Check opposite direction
            d_a2 = np.arccos(np.clip(-cos_a, -1.0, 1.0))
            d_b2 = np.arccos(np.clip(-cos_b, -1.0, 1.0))
            d_c2 = np.arccos(np.clip(-cos_c, -1.0, 1.0))
            
            r2 = np.max([d_a2, d_b2, d_c2])
            
            r = min(r1, r2)
            if r > max_radius:
                max_radius = r
                
        return max_radius

    except Exception:
        return np.pi

def construct():
    start_time = time.time()
    time_limit = 140.0 # Safety margin
    
    # Initialize with Fibonacci sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    # Repulsive force minimization
    # We use a potential V = sum 1/d^2 or similar to spread points out.
    # This is a standard Tammes problem heuristic.
    
    current_radius = compute_covering_radius(points)
    best_points = points.copy()
    best_radius = current_radius
    
    step_size = 0.05
    decay = 0.99
    
    while time.time() - start_time < time_limit:
        # Compute pairwise distances
        # D[i,j] = distance between i and j
        # We want to maximize min distance, so we minimize -min distance or sum of repulsive forces.
        
        # Vectorized repulsive force calculation
        # F_i = sum_{j != i} (points[i] - points[j]) / ||points[i] - points[j]||^3
        # This is the force for a 1/r^2 potential.
        
        # Compute all pairwise differences
        # To save memory/time, we can do it in chunks or just use full matrix since N=130 is small (130x130x3 is tiny)
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        
        # Avoid division by zero
        np.fill_diagonal(dist_sq, 1.0)
        
        # Force magnitude: 1 / dist^3 (derivative of 1/dist^2 is -2/dist^3, direction is diff)
        # We want to move points away from each other.
        # Force on i due to j is proportional to (p_i - p_j) / dist^3
        inv_dist_cubed = 1.0 / (dist_sq * np.sqrt(dist_sq))
        np.fill_diagonal(inv_dist_cubed, 0.0)
        
        forces = np.sum(diff * inv_dist_cubed[:, :, np.newaxis], axis=1)
        
        # Normalize forces to have similar magnitude
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms[force_norms == 0] = 1.0
        forces = forces / force_norms
        
        # Update points
        points = points + step_size * forces
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Decay step size
        step_size *= decay
        
        # Periodically check the actual covering radius
        # Doing this every iteration is too slow. Do it every 10 iterations.
        if int((time.time() - start_time) * 10) % 10 == 0:
             current_radius = compute_covering_radius(points)
             if current_radius < best_radius:
                 best_radius = current_radius
                 best_points = points.copy()
        
        if time.time() - start_time > time_limit - 5.0:
             break

    # Final check
    final_radius = compute_covering_radius(best_points)
    if final_radius < best_radius:
         best_radius = final_radius
    
    return best_points
