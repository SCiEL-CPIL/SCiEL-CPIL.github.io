import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _compute_circumradii(points, hull):
    """
    Compute circumradii and circumcenters for all hull facets.
    Returns:
        radii: (M,) array of circumradii (angular)
        centers: (M, 3) array of circumcenters on sphere
        facets: (M, 3) array of facet indices
    """
    facets = hull.simplices
    M = len(facets)
    
    # Get all triangle vertices at once
    p1 = points[facets[:, 0]]  # (M, 3)
    p2 = points[facets[:, 1]]  # (M, 3)
    p3 = points[facets[:, 2]]  # (M, 3)
    
    # Compute cross products
    v1 = p2 - p1  # (M, 3)
    v2 = p3 - p1  # (M, 3)
    cross = np.cross(v1, v2)  # (M, 3)
    norm_cross = np.linalg.norm(cross, axis=1)  # (M,)
    
    # Handle degenerate triangles
    valid = norm_cross > 1e-14
    centers = np.zeros((M, 3))
    radii = np.zeros(M)
    
    if np.any(valid):
        c = cross[valid] / norm_cross[valid, np.newaxis]  # (M_valid, 3)
        
        # Ensure centers point to the empty cap side
        # The centroid of each triangle should be on the same side as the center
        centroids = (p1[valid] + p2[valid] + p3[valid]) / 3.0
        dots = np.einsum('ij,ij->i', c, centroids)
        flip = dots < 0
        c[flip] = -c[flip]
        
        centers[valid] = c
        
        # Compute angular distances
        dot_vals = np.einsum('ij,ij->i', c, p1[valid])
        dot_vals = np.clip(dot_vals, -1.0, 1.0)
        radii[valid] = np.arccos(dot_vals)
    
    return radii, centers, facets

def _compute_covering_radius(points):
    """
    Compute the covering radius in degrees.
    """
    try:
        hull = ConvexHull(points)
        radii, centers, facets = _compute_circumradii(points, hull)
        return np.degrees(np.max(radii))
    except Exception:
        return 180.0

def _gradient_of_max_circumradius(points, hull, threshold=0.001):
    """
    Compute a gradient that reduces the maximum circumradius.
    For each point, compute the weighted sum of directions to circumcenters
    of facets that are near the maximum.
    
    Returns:
        gradient: (N, 3) array of gradient directions (pointing toward reduction)
    """
    radii, centers, facets = _compute_circumradii(points, hull)
    max_r = np.max(radii)
    
    # Find facets that are near the maximum
    threshold_val = max_r - threshold
    critical = radii > threshold_val
    
    if not np.any(critical):
        return np.zeros_like(points)
    
    # For each critical facet, distribute the "force" to its three vertices
    # The force on each vertex points from the vertex toward the circumcenter
    # (projected onto the tangent plane)
    
    gradient = np.zeros_like(points)
    
    for k in np.where(critical)[0]:
        c = centers[k]
        tri = facets[k]
        # Weight by how critical this facet is
        weight = (radii[k] - threshold_val) / (max_r - threshold_val + 1e-10)
        
        for idx in tri:
            p = points[idx]
            # Direction from p to c
            v = c - p
            # Project onto tangent plane at p
            dot_vp = np.dot(v, p)
            tangent_v = v - dot_vp * p
            norm_tv = np.linalg.norm(tangent_v)
            if norm_tv > 1e-14:
                gradient[idx] += weight * tangent_v / norm_tv
    
    return gradient

def construct():
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    # Add small random perturbation to break symmetry
    rng = np.random.default_rng(42)
    points += rng.normal(0, 0.01, points.shape)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Optimization parameters
    step_size = 0.02
    min_step_size = 1e-7
    no_improve_count = 0
    max_no_improve = 30
    
    # Multi-stage optimization
    stages = [
        (0.05, 0.001, 0.5),   # (step_size, threshold, time_fraction)
        (0.02, 0.0005, 0.3),
        (0.005, 0.0001, 0.2),
    ]
    
    total_time_budget = deadline - start_time
    elapsed = 0.0
    
    for step_init, thresh_init, time_frac in stages:
        stage_deadline = start_time + total_time_budget * (1.0 - time_frac) + (deadline - start_time - total_time_budget) * 0.5
        # Actually, let's just use a simpler time allocation
        stage_end = start_time + (deadline - start_time) * (1.0 - time_frac) + 10.0
        
        step_size = step_init
        threshold = thresh_init
        
        while time.time() < stage_end:
            # Compute hull
            try:
                hull = ConvexHull(points)
            except Exception:
                # If hull fails, try a random kick
                kick = rng.normal(0, 0.02, points.shape)
                points += kick
                points /= np.linalg.norm(points, axis=1, keepdims=True)
                continue
            
            # Compute gradient
            grad = _gradient_of_max_circumradius(points, hull, threshold)
            
            # Normalize gradient
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-14:
                break
            
            # Update points
            new_points = points + step_size * grad
            new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
            
            current_radius = _compute_covering_radius(new_points)
            
            if current_radius < best_radius:
                best_radius = current_radius
                best_points = new_points.copy()
                points = new_points
                no_improve_count = 0
                # Slightly increase step size if making progress
                step_size = min(step_size * 1.02, step_init * 2.0)
            else:
                # If worse, revert and decrease step size
                step_size = max(step_size * 0.9, min_step_size)
                no_improve_count += 1
                
                if no_improve_count > max_no_improve:
                    # Stagnation: try a random kick
                    kick = rng.normal(0, 0.03, points.shape)
                    points += kick
                    points /= np.linalg.norm(points, axis=1, keepdims=True)
                    step_size = step_init * 0.5
                    no_improve_count = 0
                    current_radius = _compute_covering_radius(points)
                    if current_radius < best_radius:
                        best_radius = current_radius
                        best_points = points.copy()
    
    # Final fine-tuning with very small steps
    step_size = 0.001
    threshold = 0.00005
    no_improve_count = 0
    
    while time.time() < deadline:
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        grad = _gradient_of_max_circumradius(points, hull, threshold)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-14:
            break
        
        new_points = points + step_size * grad
        new_points /= np.linalg.norm(new_points, axis=1, keepdims=True)
        
        current_radius = _compute_covering_radius(new_points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = new_points.copy()
            points = new_points
            no_improve_count = 0
        else:
            step_size = max(step_size * 0.95, min_step_size)
            no_improve_count += 1
            
            if no_improve_count > 50:
                break
    
    return best_points
