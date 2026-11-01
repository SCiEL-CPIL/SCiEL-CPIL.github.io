"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
import math

def _compute_facet_data(points):
    """Compute circumcenters and radii for all convex hull facets."""
    try:
        hull = ConvexHull(points)
    except Exception:
        return None, None
    
    num_facets = len(hull.simplices)
    centers = np.zeros((num_facets, 3))
    radii = np.zeros(num_facets)
    
    for i, simplex in enumerate(hull.simplices):
        p1, p2, p3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        v = p2 - p1
        w = p3 - p1
        vv = np.dot(v, v)
        ww = np.dot(w, w)
        vw = np.dot(v, w)
        det = vv * ww - vw * vw
        
        if abs(det) < 1e-12:
            centers[i] = (p1 + p2 + p3) / 3.0
            radii[i] = 0.0
            continue
        
        t1 = (0.5 * vv * ww - 0.5 * ww * vw) / det
        t2 = (0.5 * ww * vv - 0.5 * vv * vw) / det
        C_plane = p1 + t1 * v + t2 * w
        norm_C = np.linalg.norm(C_plane)
        
        if norm_C < 1e-12:
            centers[i] = (p1 + p2 + p3) / 3.0
            radii[i] = 0.0
            continue
            
        C = C_plane / norm_C
        # Ensure C is on the correct side (outward normal)
        # The normal to the facet is cross(v, w). 
        # The circumcenter should be on the same side as the normal.
        n = np.cross(v, w)
        if np.dot(C, n) < 0:
             C = -C
             
        cos_R = np.clip(np.dot(C, p1), -1.0, 1.0)
        R = np.arccos(cos_R)
        
        centers[i] = C
        radii[i] = R
        
    return centers, radii

def _covering_radius(points):
    """Compute the covering radius (in radians)."""
    centers, radii = _compute_facet_data(points)
    if radii is None:
        return np.pi
    return np.max(radii)

def _local_optimize(points, max_iter=2000, step_size=0.01):
    """Optimize points using vector field from facet circumcenters."""
    points = points.copy()
    for it in range(max_iter):
        centers, radii = _compute_facet_data(points)
        if radii is None:
            break
        
        max_r = np.max(radii)
        if max_r < 0.197: # ~11.3 degrees
            break
            
        # Create a force vector for each point
        forces = np.zeros_like(points)
        
        # Weight facets by their radius (larger radius -> stronger force)
        # Use a threshold to focus on the worst facets
        threshold = np.percentile(radii, 80) # Focus on top 20% worst facets
        
        for i, simplex in enumerate(ConvexHull(points).simplices):
            if radii[i] < threshold:
                continue
                
            C = centers[i]
            # Add force to each vertex of this facet
            for idx in simplex:
                p = points[idx]
                # Direction from p to C
                direction = C - p
                # Project direction onto tangent plane at p
                # Tangent plane normal is p.
                # Projection: d - (d.p)p
                proj = direction - np.dot(direction, p) * p
                forces[idx] += proj * radii[i] # Weight by radius
        
        # Normalize forces and apply
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms[force_norms == 0] = 1.0
        normalized_forces = forces / force_norms
        
        # Adaptive step size? Keep it simple for now.
        points = points + step_size * normalized_forces
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
    return points

def construct():
    best_points = None
    best_radius = np.pi
    
    # Multiple restarts
    for restart in range(5):
        # Initialize with Fibonacci spiral + random perturbation
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
        if restart > 0:
            # Add random perturbation
            noise = np.random.normal(0, 0.1, (N, 3))
            points = points + noise
            points = points / np.linalg.norm(points, axis=1, keepdims=True)
            
        # Optimize
        optimized_points = _local_optimize(points)
        
        # Evaluate
        radius = _covering_radius(optimized_points)
        
        if radius < best_radius:
            best_radius = radius
            best_points = optimized_points.copy()
            
    return best_points
# EVOLVE-BLOCK-END
