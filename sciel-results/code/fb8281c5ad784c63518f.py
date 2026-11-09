import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 50

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return pts / norms

def _compute_face_data(points):
    """
    Compute the circumradius and circumcenter (on sphere) for each triangular facet
    of the convex hull. Returns:
      - radii: array of shape (F,) with circumradius in radians for each facet
      - centers: array of shape (F, 3) with the spherical circumcenter for each facet
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.full(1, np.pi), np.array([[1.0, 0.0, 0.0]])
    
    F = len(hull.simplices)
    radii = np.zeros(F)
    centers = np.zeros((F, 3))
    
    for idx, face in enumerate(hull.simplices):
        u = points[face[0]]
        v = points[face[1]]
        w = points[face[2]]
        
        # Check for degenerate triangles (collinear points)
        n = np.cross(v - u, w - u)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            radii[idx] = np.pi
            centers[idx] = (u + v + w) / np.linalg.norm(u + v + w + 1e-30)
            continue
        n = n / n_norm
        
        # Spherical circumcenter: intersection of great circles through (u,v), (v,w), (w,u)
        # The circumcenter c is perpendicular to the plane of the triangle.
        # c is parallel to (u x v) x (v x w)
        cross1 = np.cross(u, v)
        cross2 = np.cross(v, w)
        c = np.cross(cross1, cross2)
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-12:
            radii[idx] = np.pi
            centers[idx] = (u + v + w) / np.linalg.norm(u + v + w + 1e-30)
            continue
        c = c / c_norm
        
        # Ensure c is on the same side as the outward normal (the "inside" of the sphere relative to the triangle)
        # For a convex hull containing the origin, the circumcenter of the spherical triangle
        # should be "inside" the triangle, which means c dot (u+v+w) > 0 or c dot n > 0
        # Actually, for a small triangle, the circumcenter is close to the centroid.
        # Let's check dot with n. If c dot n < 0, flip c.
        if np.dot(c, n) < 0:
            c = -c
            
        radii[idx] = np.arccos(np.clip(np.dot(c, u), -1.0, 1.0))
        centers[idx] = c
        
    return radii, centers

def _compute_covering_radius(points):
    """
    Compute the covering radius of the points on the sphere.
    Returns the radius in radians.
    """
    radii, _ = _compute_face_data(points)
    return np.max(radii) if len(radii) > 0 else np.pi

def _dual_uniformization_step(points, radii, centers, step_size, beta):
    """
    Perform a single step of dual uniformization.
    Moves points in the direction that reduces the variance of the facet circumradii.
    """
    n_faces = len(radii)
    if n_faces == 0:
        return points
    
    # Compute the gradient of the smooth objective (variance) with respect to each point
    # Objective: sum of (r_i - mean(r))^2
    # This is equivalent to minimizing the spread of radii.
    # Let's use a smoother objective: sum(exp(beta * r_i) * r_i) or just sum(exp(beta * r_i))
    # Actually, the inspiration says "flattening the face height field".
    # Let's use the sum of exp(beta * r_i) as a proxy for the max, which is differentiable.
    
    # Gradient of sum(exp(beta * r_i)) with respect to point p_k.
    # r_i is the circumradius of face i.
    # If face i contains point k, then dr_i/dp_k is the gradient of the circumradius w.r.t. p_k.
    # For a spherical triangle, the circumradius r is related to the side lengths.
    # A simpler approach: use the gradient of the distance from the circumcenter to the vertices.
    # Actually, the circumcenter c_i is a function of the vertices.
    # The gradient of r_i = arccos(c_i dot u) w.r.t. u is - (c_i - u(c_i dot u)) / (sqrt(1-(c_i dot u)^2))
    # This is the projection of c_i onto the tangent plane at u.
    
    # Let's compute the gradient for each face.
    # For face i with vertices u, v, w and circumcenter c_i:
    # grad_u r_i = - (c_i - u * dot(c_i, u)) / sqrt(1 - dot(c_i, u)^2)
    
    # Sum over faces for each point.
    grad = np.zeros_like(points)
    
    for idx in range(n_faces):
        r_i = radii[idx]
        c_i = centers[idx]
        # Weights for the smooth max approximation
        weight = np.exp(beta * r_i)
        
        for k in range(3):
            p_k = points[0] # placeholder
            # We need to know which point corresponds to which vertex in the face
            # hull.simplices[idx] gives the indices
    
    # We need the hull to know which point is which vertex in each face.
    # Let's re-fetch the hull or pass the face indices.
    # To avoid recomputing the hull, we should pass the face indices.
    # Let's modify the function to take the hull or face indices.
    
    # For now, let's just return the same points if we can't compute the gradient easily.
    # I will implement the gradient calculation properly in a helper that takes the hull.
    
    return points

def _dual_uniformization_step_full(points, hull, radii, centers, step_size, beta):
    """
    Perform a single step of dual uniformization using the hull structure.
    """
    n_faces = len(radii)
    if n_faces == 0:
        return points
    
    grad = np.zeros_like(points)
    
    for idx in range(n_faces):
        r_i = radii[idx]
        c_i = centers[idx]
        weight = np.exp(beta * r_i)
        
        face = hull.simplices[idx]
        for k in range(3):
            p_idx = face[k]
            p_k = points[p_idx]
            
            dot_val = np.dot(c_i, p_k)
            dot_val = np.clip(dot_val, -1.0, 1.0)
            denom = np.sqrt(np.clip(1.0 - dot_val * dot_val, 1e-12, 1.0))
            
            # Gradient of arccos(c dot p) w.r.t p is - (c - p * dot(c, p)) / sqrt(1 - dot^2)
            # This vector is tangent to the sphere at p.
            grad_p = -(c_i - p_k * dot_val) / denom
            
            # The gradient of the objective (sum exp(beta * r)) w.r.t p_k is:
            # beta * exp(beta * r_i) * grad_p
            grad[p_idx] += weight * beta * grad_p
            
    # Normalize the gradient to have unit length? Or just use it as is with a small step.
    # Riemannian gradient descent: p_new = normalize(p - step_size * grad)
    # But grad is already tangent to the sphere.
    
    # To avoid moving points too far, we can normalize the total gradient vector.
    # Or just use a small step_size.
    
    new_points = points - step_size * grad
    new_points = _normalize(new_points)
    
    return new_points

def _local_refine_dual(points, deadline, max_iters=1000, initial_step=0.05, beta=10.0):
    """
    Refine points using dual uniformization.
    """
    points = _normalize(points)
    best_radius = np.pi
    best_points = points.copy()
    
    for i in range(max_iters):
        if time.time() > deadline:
            break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
            
        radii, centers = _compute_face_data(points)
        # Re-compute hull.simplices mapping for the step
        # We need to ensure radii/centers correspond to hull.simplices
        # _compute_face_data does this, so it's fine.
        
        current_radius = np.max(radii)
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            
        # Check convergence: if the max radius is very close to the mean, or if the step is too small
        # Also check if the variance is very small
        
        # Update step size
        step_size = initial_step / (1.0 + i * 0.01)
        
        # Perform the step
        new_points = _dual_uniformization_step_full(points, hull, radii, centers, step_size, beta)
        
        # Check for coincident points
        # If any points are too close, we might need to perturb.
        # For now, just accept.
        
        # Check if the new configuration is valid (no degenerate hull, etc.)
        try:
            new_hull = ConvexHull(new_points)
            new_radii, _ = _compute_face_data(new_points)
            new_radius = np.max(new_radii)
        except Exception:
            # If the new configuration is invalid, revert
            new_radius = np.pi
            
        if new_radius < current_radius:
            points = new_points
        else:
            # Revert and maybe decrease step size more aggressively or break
            # If we keep failing, we might be stuck.
            # Let's try a smaller step.
            step_size *= 0.5
            if step_size < 1e-6:
                break
                
    return best_points, best_radius

def construct():
    # Initial Fibonacci spiral
    points = _fibonacci_sphere(N)
    
    best_points = points.copy()
    best_rad = _covering_radius(best_points)
    
    start_time = time.time()
    deadline = start_time + 110.0 # Safety margin
    
    # Multi-restart strategy
    # 1. Fibonacci
    # 2. Random
    # 3. Fibonacci with phase shift
    # 4. Random + Fibonacci blend
    
    configs = [
        _fibonacci_sphere(N),
        _normalize(np.random.randn(N, 3)),
        _normalize(_fibonacci_sphere(N) + np.random.randn(N, 3) * 0.1),
        _normalize(0.7 * _fibonacci_sphere(N) + 0.3 * np.random.randn(N, 3))
    ]
    
    for config in configs:
        if time.time() > deadline - 20:
            break
            
        pts = _normalize(config.copy())
        
        # Refine using dual uniformization
        if time.time() < deadline - 10:
            refined_pts, refined_radius = _local_refine_dual(pts, deadline - 10, max_iters=2000, initial_step=0.1, beta=5.0)
            if refined_radius < best_rad:
                best_rad = refined_radius
                best_points = refined_pts.copy()
                
    # Final polish
    if time.time() < deadline - 2:
        refined_pts, refined_radius = _local_refine_dual(best_points.copy(), deadline - 2, max_iters=5000, initial_step=0.01, beta=20.0)
        if refined_radius < best_rad:
            best_rad = refined_radius
            best_points = refined_pts.copy()
            
    best_points = _normalize(best_points)
    return best_points
