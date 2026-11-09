import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _compute_covering_radius(points):
    """
    Compute the covering radius of the point set on the unit sphere.
    Returns the max angular distance in radians.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        # If hull fails, fallback to a simple check or return a large value
        # This can happen if points are coplanar or nearly so
        return np.pi
        
    faces = hull.simplices
    max_r = 0.0
    for face in faces:
        p1, p2, p3 = points[face[0]], points[face[1]], points[face[2]]
        # Compute circumcenter of the spherical triangle
        # Normal to the plane
        n = np.cross(p2 - p1, p3 - p1)
        if np.linalg.norm(n) < 1e-10:
            # Degenerate triangle
            # Calculate max distance to the 3 points from their centroid?
            # Or just use max distance between points
            d12 = np.arccos(np.clip(np.dot(p1, p2), -1, 1))
            d23 = np.arccos(np.clip(np.dot(p2, p3), -1, 1))
            d31 = np.arccos(np.clip(np.dot(p3, p1), -1, 1))
            # Rough estimate for degenerate case
            r = max(d12, d23, d31) / 2.0 
            max_r = max(max_r, r)
            continue
            
        c = n / np.linalg.norm(n)
        
        # Determine correct sign: c should be such that c . p1 > 0 (for small caps)
        # If c . p1 < 0, the center is on the other side
        if np.dot(c, p1) < 0:
            c = -c
            
        # Calculate radius
        cos_r = np.dot(c, p1)
        cos_r = np.clip(cos_r, -1.0, 1.0)
        r = np.arccos(cos_r)
        max_r = max(max_r, r)
        
    return max_r

def _get_gradients(points):
    """
    Compute the soft-max weighted gradients for the covering radius.
    Returns a list of gradients for each point.
    """
    try:
        hull = ConvexHull(points)
    except Exception:
        return [np.zeros(3) for _ in range(N)]
        
    faces = hull.simplices
    n_faces = len(faces)
    
    # Compute radii and centers for all faces
    centers = []
    radii = []
    for face in faces:
        p1, p2, p3 = points[face[0]], points[face[1]], points[face[2]]
        n = np.cross(p2 - p1, p3 - p1)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            centers.append(np.zeros(3))
            radii.append(np.pi) # Large penalty
            continue
        c = n / norm_n
        if np.dot(c, p1) < 0:
            c = -c
        cos_r = np.clip(np.dot(c, p1), -1.0, 1.0)
        r = np.arccos(cos_r)
        centers.append(c)
        radii.append(r)
        
    radii = np.array(radii)
    
    # Soft-max weights
    # Lambda controls the "sharpness". Higher lambda focuses more on the worst face.
    # Start with moderate lambda.
    lam = 10.0 
    exp_r = np.exp(lam * radii)
    sum_exp = np.sum(exp_r)
    if sum_exp == 0:
        weights = np.ones(n_faces) / n_faces
    else:
        weights = exp_r / sum_exp
        
    grads = [np.zeros(3) for _ in range(N)]
    
    for k, face in enumerate(faces):
        w = weights[k]
        if w < 1e-8:
            continue
        c = centers[k]
        r = radii[k]
        sin_r = np.sin(r)
        if sin_r < 1e-8:
            continue
            
        for idx in face:
            p = points[idx]
            cos_r = np.dot(c, p)
            # Gradient of arccos(c.p) w.r.t p
            # d/dp arccos(c.p) = -(c - (c.p)p) / sqrt(1-(c.p)^2)
            # Note: This is the gradient in R^3. We project to tangent space by renormalizing later.
            # The formula (c - cos_r * p) / sin_r is already tangent to the sphere at p.
            grad = (c - cos_r * p) / sin_r
            grads[idx] += w * grad
            
    return grads

def construct():
    start_time = time.time()
    deadline = 110.0 # 10s buffer
    
    # Initialize
    points = _fibonacci_sphere(N)
    
    # Phase 1: Gradient Descent
    # Adaptive learning rate
    eta = 0.01
    lam = 5.0
    
    best_radius = _compute_covering_radius(points)
    best_points = points.copy()
    
    for i in range(5000):
        if time.time() - start_time > deadline:
            break
            
        # Compute gradients
        grads = _get_gradients(points)
        
        # Update points
        for j in range(N):
            # Move in direction of negative gradient (minimize radius)
            # Note: _get_gradients returns gradient of radius. We want to decrease it.
            new_p = points[j] - eta * grads[j]
            norm = np.linalg.norm(new_p)
            if norm > 1e-10:
                points[j] = new_p / norm
            else:
                points[j] = np.array([0, 0, 1])
                
        # Calculate current radius
        current_radius = _compute_covering_radius(points)
        
        if current_radius < best_radius:
            best_radius = current_radius
            best_points = points.copy()
            # Slow down if improving
            eta *= 0.99
        else:
            # Revert to best if it got worse significantly? Or just keep going?
            # For simplicity, keep going but maybe increase eta if stuck?
            # Let's just keep the current points and continue.
            # However, if it's much worse, revert to best_points to avoid divergence.
            if current_radius > best_radius * 1.1:
                points = best_points.copy()
                eta *= 0.5
                
        # Adjust lambda to focus more on worst cases as we converge
        lam = 5.0 + i * 0.01
        
    # Phase 2: Polish with Nelder-Mead
    if time.time() - start_time < deadline - 10:
        def objective(x_flat):
            pts = x_flat.reshape(N, 3)
            # Normalize
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            pts = pts / norms
            return _compute_covering_radius(pts)
        
        x0 = best_points.flatten()
        
        # Use Nelder-Mead for local optimization
        # This might be slow, so we limit iterations
        from scipy.optimize import minimize
        
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10})
            if res.success:
                new_pts = res.x.reshape(N, 3)
                norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
                new_pts = new_pts / norms
                new_radius = _compute_covering_radius(new_pts)
                if new_radius < best_radius:
                    best_points = new_pts
                    best_radius = new_radius
        except Exception:
            pass
            
    return best_points
