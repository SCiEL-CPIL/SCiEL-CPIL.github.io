import numpy as np
import scipy.spatial
import time
from scipy.optimize import minimize

N = 130

def _covering_radius_from_hull(pts):
    """Compute exact covering radius in degrees from convex hull facets."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    try:
        hull = scipy.spatial.ConvexHull(pts)
    except:
        return 1000.0, -1, None, None
        
    max_rad = 0.0
    worst_idx = -1
    worst_circumcenter = None
    worst_simplex = None
    
    for idx, simplex in enumerate(hull.simplices):
        v1, v2, v3 = pts[simplex]
        n = np.cross(v1 - v2, v1 - v3)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        n = n / n_norm
        
        if n.dot(v1) < 0:
            n = -n
            
        d = np.clip(n.dot(v1), -1.0, 1.0)
        rad = np.arccos(d)
        
        if rad > max_rad:
            max_rad = rad
            worst_idx = idx
            worst_circumcenter = n
            worst_simplex = simplex
            
    return np.degrees(max_rad), worst_idx, worst_circumcenter, worst_simplex

def _soft_covering_radius_and_grad(pts, beta=50.0):
    """
    Compute a smooth surrogate for the covering radius using log-sum-exp over facets.
    Returns (value, gradient) where value is in degrees.
    """
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    
    try:
        hull = scipy.spatial.ConvexHull(pts)
    except:
        return 1000.0, np.zeros_like(pts)
    
    # Collect all facet normals and their circumradius cosines
    normals = []
    cos_dists = []
    simplex_indices = []
    
    for idx, simplex in enumerate(hull.simplices):
        v1, v2, v3 = pts[simplex]
        n = np.cross(v1 - v2, v1 - v3)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        n = n / n_norm
        if n.dot(v1) < 0:
            n = -n
        d = np.clip(n.dot(v1), -1.0, 1.0)
        normals.append(n)
        cos_dists.append(d)
        simplex_indices.append(simplex)
    
    if len(normals) == 0:
        return 1000.0, np.zeros_like(pts)
    
    normals = np.array(normals)  # (F, 3)
    cos_dists = np.array(cos_dists)  # (F,)
    
    # Convert to radians: rad = arccos(d)
    # We want to maximize the minimum... no, minimize the maximum arccos(d)
    # Let's work with the angles in radians
    angles = np.arccos(cos_dists)  # (F,)
    
    # Soft max via log-sum-exp
    # soft_max(angles) = (1/beta) * log(sum(exp(beta * angles)))
    # For numerical stability, subtract max
    max_angle = np.max(angles)
    exp_angles = np.exp(beta * (angles - max_angle))
    log_sum = np.log(np.sum(exp_angles))
    soft_rad = max_angle + log_sum / beta
    
    # Gradient of soft_rad with respect to each point
    # soft_rad = max_angle + (1/beta) * log(sum_j exp(beta * (angle_j - max_angle)))
    # d(soft_rad)/d(angle_k) = exp(beta*(angle_k - max_angle)) / sum_j exp(beta*(angle_j - max_angle))
    #                        = exp_angles[k] / sum(exp_angles)
    weights = exp_angles / np.sum(exp_angles)  # (F,)
    
    # Now d(angle_k)/d(pts)
    # angle_k = arccos(d_k) where d_k = n_k . v1_k
    # d(angle_k)/d(d_k) = -1/sqrt(1-d_k^2)
    # d(d_k)/d(pts) involves the gradient of the normal and the vertex
    
    # This is complex. Let's use a simpler approach:
    # The gradient of the covering radius w.r.t. point i is approximately
    # the weighted sum of gradients from facets that include point i.
    
    # For a facet with normal n and vertex v (one of the simplex vertices),
    # d(rad)/d(v) = -n / sqrt(1 - (n.v)^2) * (component of v change perpendicular to n)
    # Actually, let's use the fact that for a small perturbation dv of vertex v,
    # the change in d = n.v is n.dv
    # d(rad) = -dv_d / sqrt(1-d^2) where dv_d = n.dv
    # So d(rad)/d(v) = -n / sqrt(1-d^2)
    
    grad = np.zeros_like(pts)
    
    for f_idx in range(len(normals)):
        if weights[f_idx] < 1e-10:
            continue
        n = normals[f_idx]
        d = cos_dists[f_idx]
        sqrt_term = np.sqrt(np.maximum(1.0 - d*d, 1e-10))
        
        # Gradient contribution from this facet to its vertices
        simplex = simplex_indices[f_idx]
        for vi in simplex:
            v = pts[vi]
            # d(rad_f)/d(v) = -n / sqrt(1-d^2)
            # But we need to project onto the tangent plane of the sphere
            # The tangent projection of -n at v is -n + (n.v)v = -n + d*v
            tangent_grad = -n + d * v
            grad[vi] += weights[f_idx] * tangent_grad / sqrt_term
    
    # Convert from radians to degrees
    soft_rad_deg = np.degrees(soft_rad)
    grad_deg = grad * (180.0 / np.pi)
    
    return soft_rad_deg, grad_deg

def _lloyd_step(pts):
    """
    Perform a Lloyd iteration: move each point to the centroid of its Voronoi cell.
    On the sphere, we use the spherical centroid.
    """
    # Use Delaunay triangulation on the sphere (equivalent to Voronoi)
    try:
        tri = scipy.spatial.Delaunay(pts)
    except:
        return pts
    
    new_pts = pts.copy()
    
    for i in range(N):
        # Find all triangles that include point i
        # Sum up the vertices of those triangles, weighted by triangle area (or just count)
        # For a proper spherical Lloyd, we should compute the centroid of the Voronoi cell
        # A simple approximation: average the neighbors
        
        # Find neighbors: points that share a triangle with i
        neighbors = set()
        for simplex in tri.simplices:
            if i in simplex:
                for j in simplex:
                    if j != i:
                        neighbors.add(j)
        
        if len(neighbors) == 0:
            continue
            
        # Compute the spherical centroid: average of neighbor points, then normalize
        # A better way: use the centroid of the Voronoi cell
        # For now, use a simple weighted average of neighbors
        neighbor_pts = pts[list(neighbors)]
        centroid = np.mean(neighbor_pts, axis=0)
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm > 1e-10:
            centroid = centroid / centroid_norm
            # Blend with original point to avoid large jumps
            new_pts[i] = 0.5 * pts[i] + 0.5 * centroid
            new_pts[i] = new_pts[i] / np.linalg.norm(new_pts[i])
    
    return new_pts

def _exact_local_optimize(pts, deadline, max_iter=500):
    """
    Refine the configuration by targeting the worst facet and moving its vertices.
    """
    current_radius, _, _, _ = _covering_radius_from_hull(pts)
    alpha = 0.05
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        current_radius, worst_idx, worst_circumcenter, worst_simplex = _covering_radius_from_hull(pts)
        
        if worst_simplex is None:
            break
            
        v1, v2, v3 = pts[worst_simplex]
        c = worst_circumcenter
        
        # Move v1, v2, v3 towards c
        new_points = pts.copy()
        
        for i in worst_simplex:
            v = pts[i]
            proj = c - np.dot(v, c) * v
            new_v = v + alpha * proj
            new_points[i] = new_v / np.linalg.norm(new_v)
        
        new_radius, _, _, _ = _covering_radius_from_hull(new_points)
        
        if new_radius < current_radius:
            pts = new_points
            current_radius = new_radius
            alpha = min(alpha * 1.05, 0.1)  # Slightly increase step
        else:
            alpha *= 0.5
            if alpha < 0.001:
                # Try random perturbation
                for i in worst_simplex:
                    noise = np.random.normal(0, 0.02, 3)
                    new_v = pts[i] + noise
                    new_points[i] = new_v / np.linalg.norm(new_v)
                new_radius, _, _, _ = _covering_radius_from_hull(new_points)
                if new_radius < current_radius:
                    pts = new_points
                    current_radius = new_radius
                alpha = 0.05
    
    return pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s margin before 150s limit
    
    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Lloyd iterations to find a good global configuration
    for it in range(30):
        if time.time() > deadline - 60:
            break
        points = _lloyd_step(points)
    
    # 3. Gradient-based optimization using soft covering radius
    # Use L-BFGS-B on the flattened point array
    # But we need to handle the sphere constraint
    # Instead, use a custom gradient descent with normalization
    
    beta_schedule = [20.0, 50.0, 100.0, 200.0]
    
    for beta in beta_schedule:
        if time.time() > deadline - 30:
            break
            
        # Run a few gradient descent steps
        lr = 0.01
        for step in range(100):
            if time.time() > deadline - 30:
                break
                
            val, grad = _soft_covering_radius_and_grad(points, beta)
            
            # Gradient descent step
            new_points = points - lr * grad
            # Normalize back to sphere
            norms = np.linalg.norm(new_points, axis=1, keepdims=True)
            new_points = new_points / norms
            
            # Check if improved
            new_val, _ = _soft_covering_radius_and_grad(new_points, beta)
            if new_val < val:
                points = new_points
            else:
                lr *= 0.5
                if lr < 1e-6:
                    break
    
    # 4. Exact local optimization targeting worst facets
    points = _exact_local_optimize(points, deadline, max_iter=200)
    
    # 5. Final polish: more Lloyd iterations
    for it in range(20):
        if time.time() > deadline:
            break
        points = _lloyd_step(points)
    
    # 6. Final exact optimization
    points = _exact_local_optimize(points, deadline, max_iter=100)
    
    # Ensure points are on the unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    return points
