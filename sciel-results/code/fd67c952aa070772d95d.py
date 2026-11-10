import numpy as np
from scipy.spatial import ConvexHull
import time

N = 50

def _normalize(v):
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        return np.zeros(3)
    return v / norm

def _get_circumcenters(points, hull):
    """
    Calculate circumcenters and radii for all facets of the convex hull.
    Returns:
        centers: (F, 3) array of circumcenters (not necessarily unit norm, but on the axis)
        radii: (F,) array of circumradii (angular distance)
        faces: (F, 3) array of vertex indices
    """
    faces = hull.simplices
    F = len(faces)
    centers = np.zeros((F, 3))
    radii = np.zeros(F)
    
    for i in range(F):
        v1, v2, v3 = points[faces[i]]
        
        # Calculate circumcenter of the triangle v1, v2, v3
        # The circumcenter c lies in the plane defined by v1, v2, v3.
        # c = v1 + a*(v2-v1) + b*(v3-v1)
        # |c-v1|^2 = |c-v2|^2 = |c-v3|^2 = R^2
        
        u = v2 - v1
        w = v3 - v1
        
        A = np.array([[np.dot(u, u), np.dot(w, u)], 
                      [np.dot(u, w), np.dot(w, w)]])
        B = np.array([np.dot(u, u), np.dot(w, w)])
        
        try:
            coeffs = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            # Degenerate triangle, skip or assign large radius
            radii[i] = np.pi
            centers[i] = np.zeros(3)
            continue
            
        a, b = coeffs
        c = v1 + a * u + b * w
        
        centers[i] = c
        R_sq = np.dot(c - v1, c - v1)
        # Angular radius
        # cos(R) = dot(c, v1) / |c|
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-10:
            radii[i] = np.pi
        else:
            cos_R = np.dot(c, v1) / c_norm
            cos_R = np.clip(cos_R, -1.0, 1.0)
            radii[i] = np.arccos(cos_R)
            
    return centers, radii, faces

def _gradient_descent(points, max_iters=200, beta=10.0, eta_start=0.05, eta_decay=0.99):
    """
    Perform gradient descent on the soft-max approximation of the covering radius.
    """
    points = points.copy()
    eta = eta_start
    
    for _ in range(max_iters):
        try:
            hull = ConvexHull(points)
        except Exception:
            # If hull fails, perturb and continue
            points += np.random.normal(0, 0.01, points.shape)
            points = np.array([_normalize(p) for p in points])
            continue
            
        centers, radii, faces = _get_circumcenters(points, hull)
        
        # Softmax weights
        # Use radii^2 to emphasize larger radii more
        r_sq = radii**2
        # Shift to prevent underflow/overflow
        r_sq_shifted = r_sq - np.max(r_sq)
        weights = np.exp(beta * r_sq_shifted)
        weights /= np.sum(weights)
        
        # Compute gradients
        # grad[v] accumulates the gradient for vertex v
        grad = np.zeros_like(points)
        
        for i in range(len(faces)):
            w = weights[i]
            if w < 1e-10:
                continue
                
            c = centers[i]
            R = radii[i]
            
            if R < 1e-10:
                continue
                
            # Gradient of R w.r.t vertex v
            # We want to decrease R.
            # Direction from v to c projected on tangent plane at v.
            # g_v = (c - (v.dot(c))v)
            # This vector points towards c.
            # We subtract eta * grad, so we move in -grad direction?
            # No, we want to move TOWARDS c to reduce R.
            # So the "gradient" of the loss (which we want to minimize) is -direction_to_c.
            # Or simply: update = -eta * weight * direction_to_c?
            # Let's check: if we move v towards c, R decreases.
            # So we add eta * weight * direction_to_c to v.
            
            for v_idx in faces[i]:
                v = points[v_idx]
                # Project c onto tangent plane at v
                # v is unit vector.
                # proj = c - (v.dot(c))v
                v_dot_c = np.dot(v, c)
                tangent_dir = c - v_dot_c * v
                
                # Normalize tangent direction to get unit vector?
                # The magnitude of the effect depends on distance.
                # Using the raw vector is a common heuristic for this type of force model.
                # However, to be scale invariant, we might normalize.
                # Let's use the raw vector scaled by 1/R to make it a gradient of R.
                # grad_R = tangent_dir / R ?
                
                # Let's just use the tangent direction.
                grad[v_idx] += w * tangent_dir / R

        # Update points
        # Move in the direction of the accumulated gradient (towards circumcenters)
        points = points + eta * grad
        
        # Normalize
        points = np.array([_normalize(p) for p in points])
        
        eta *= eta_decay

    return points

def construct():
    start_time = time.time()
    max_time = 115.0
    
    best_points = None
    best_radius = np.pi
    
    np.random.seed(42)
    
    # Number of restarts
    # Each restart does a full gradient descent run
    restarts = 0
    while time.time() - start_time < max_time:
        restarts += 1
        
        # Initialization
        if restarts % 2 == 1:
            # Fibonacci Sphere
            i = np.arange(N) + 0.5
            z = 1.0 - 2.0 * i / N
            phi = np.pi * (1.0 + np.sqrt(5.0)) * i
            r = np.sqrt(1.0 - z * z)
            points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
            # Add small noise to break symmetry
            points += np.random.normal(0, 0.01, (N, 3))
            points = np.array([_normalize(p) for p in points])
        else:
            # Random
            points = np.random.randn(N, 3)
            points = np.array([_normalize(p) for p in points])
            
        # Run Gradient Descent
        # Time budget per restart
        elapsed = time.time() - start_time
        remaining = max_time - elapsed
        if remaining < 5:
            break
            
        # Estimate iterations based on time? 
        # Or just fix a number of iterations.
        # 200 iterations of GD is fast.
        # Let's do 500 iterations to ensure convergence.
        refined_points = _gradient_descent(points, max_iters=500)
        
        # Evaluate
        try:
            hull = ConvexHull(refined_points)
            centers, radii, faces = _get_circumcenters(refined_points, hull)
            radius = np.max(radii)
        except Exception:
            radius = np.pi
            
        if radius < best_radius:
            best_radius = radius
            best_points = refined_points.copy()
            
        # Early stopping if very good
        if best_radius < 0.32: # 18.3 deg is 0.3194 rad
             # If we beat the known record or get close, we might stop
             # But let's try to improve further
             pass

    if best_points is None:
        # Fallback
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    return best_points
