import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130
_EPS = 1e-12
DEADLINE = 145.0  # Safety margin before 150s kill

def _initial_fib():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, None))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_init(rng):
    """Generate random initial points."""
    pts = rng.standard_normal((N, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _compute_covering_radius(pts):
    """
    Compute the true covering radius using the convex hull.
    Returns the radius in radians.
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        
        # For each face, compute the circumradius (max angular distance to vertices)
        # The covering radius is the max over all faces of their circumradius.
        # Note: For a triangle on a sphere, the circumcenter is the normalized cross product 
        # of the edge vectors (if the triangle is small enough and doesn't contain the origin 
        # in its convex hull projection, which is guaranteed for valid hulls of points on sphere 
        # unless they are antipodal).
        
        # Actually, a more robust way to compute the max distance from a point on the sphere 
        # to the nearest site is to check the poles of the Delaunay triangles (hull faces).
        # The pole of a face is the circumcenter of that spherical triangle.
        # The distance from the pole to any vertex of the face is the circumradius.
        
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        # Compute normals (poles)
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        
        # Filter out degenerate faces (shouldn't happen with ConvexHull on unique points usually, but safe)
        valid = norms.flatten() > _EPS
        if not np.any(valid):
            return np.pi
            
        normals = cross[valid] / norms[valid]
        
        # Ensure normals point outward (away from origin, towards the face)
        # The centroid of the face should have positive dot product with the normal
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1)
        normals = normals * np.sign(signs)[:, np.newaxis]
        
        # The circumradius is the angular distance from the pole to any of the 3 vertices.
        # Due to symmetry, it's the same for all 3. We just need to pick one, e.g., A.
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        
        return np.max(radii)
        
    except Exception:
        return np.pi

def _soft_max_loss_and_grad(pts, temp=0.1):
    """
    Compute the soft-max surrogate loss and its gradient.
    Loss = log(sum(exp((d_i / temp)^2))) where d_i is the angular distance to the nearest point.
    
    To make this differentiable and computationally feasible, we approximate the "nearest point"
    distance using the poles of the convex hull. The covering radius is the max circumradius.
    We want to minimize the max circumradius.
    
    A better surrogate: 
    Let R_k be the circumradius of the k-th facet.
    We want to minimize max(R_k).
    Surrogate: L = log(sum(exp((R_k / temp)^2)))
    
    Gradient of L w.r.t. points:
    dL/dR_k = (2 R_k / temp^2) * exp((R_k/temp)^2) / sum(exp((R_j/temp)^2))
    dR_k/dP_i = gradient of circumradius of facet k w.r.t. point i.
    
    For a facet with vertices A, B, C and pole N:
    R = arccos(N . A)
    dR/dN = -A / sqrt(1 - (N.A)^2)
    dR/dA = -N / sqrt(1 - (N.A)^2)
    
    This is complex to vectorize perfectly for all points. 
    Alternative simpler strategy: 
    Instead of full gradient descent on the hull structure (which changes topology),
    we use a "Push" strategy based on the current hull.
    
    However, the prompt asks for a MATERIAL DIFFERENCE. 
    Let's use a simpler, robust gradient flow:
    1. Compute Hull.
    2. Identify worst facets.
    3. Move points towards the circumcenters of the facets they belong to, 
       weighted by the "pressure" (how much larger the facet is compared to average).
    
    Actually, let's implement a proper Gradient Descent on the sphere using the 
    "Voronoi Cell Centroid" approach but with a continuous step size and 
    adaptive temperature for the soft-max over the *distances to the poles*.
    
    Wait, the current program uses Lloyd's. 
    Let's try **Gradient Descent on the Soft-Max of Facet Radii**.
    
    For each point i, it belongs to a set of facets F_i.
    The objective is to minimize the max radius.
    We approximate the gradient of the max radius by summing the gradients of the 
    radii of the facets incident to point i, weighted by their contribution to the max.
    
    For a facet f with vertices v1, v2, v3 and pole p:
    R_f = arccos(p . v1)
    
    If we move vertex v1 by delta, R_f changes.
    dR_f/dv1 = -p / sin(R_f)  (vector on sphere, tangent to sphere at v1)
    
    So the gradient contribution to point i from facet f is:
    weight_f * (-p_f / sin(R_f)) projected onto the tangent plane at pts[i].
    
    weight_f = exp((R_f/temp)^2) / sum(exp((R_j/temp)^2))
    """
    try:
        hull = ConvexHull(pts)
        faces = hull.simplices
        
        A = pts[faces[:, 0]]
        B = pts[faces[:, 1]]
        C = pts[faces[:, 2]]
        
        AB = B - A
        AC = C - A
        cross = np.cross(AB, AC)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        
        valid = norms.flatten() > _EPS
        if not np.any(valid):
            return 0.0, np.zeros_like(pts)
            
        normals = cross[valid] / norms[valid]
        centroids = (A[valid] + B[valid] + C[valid]) / 3.0
        signs = np.sum(normals * centroids, axis=1)
        normals = normals * np.sign(signs)[:, np.newaxis]
        
        valid_faces = faces[valid]
        
        # Compute radii for valid faces
        cos_theta = np.sum(normals * A[valid], axis=1)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        radii = np.arccos(cos_theta)
        
        # Soft-max weights
        # Scale radii to be around 1 for numerical stability in exp
        # Typical radius ~ 0.2 rad (11 deg). 
        # temp controls the "sharpness". 
        # If temp is small, only the worst facets matter.
        # If temp is large, all facets matter (Lloyd-like).
        
        # Let's use a dynamic temperature or a fixed one that emphasizes the max.
        # We want to minimize the max, so we want weights concentrated on the max.
        # But we also want to balance. 
        # Let's use temp = 0.05 rad (approx 3 degrees).
        temp = 0.05 
        
        # Normalize radii by the max to keep exp arguments bounded?
        # Or just use raw radii. 
        # exp((0.2/0.05)^2) = exp(16) ~ 9e6. exp((0.19/0.05)^2) = exp(14.4) ~ 1.8e6.
        # This is stable enough.
        
        scaled_radii = radii / temp
        exp_vals = np.exp(scaled_radii**2)
        sum_exp = np.sum(exp_vals)
        weights = exp_vals / sum_exp
        
        # Gradient accumulation
        # grad[i] += weights[f] * dR_f/dpts[i]
        # For a face f with vertices [i, j, k] and pole p, radius R:
        # dR/dpts[i] = -p / sin(R)  (this is a vector in R3, needs projection to tangent space)
        # Actually, the gradient on the sphere is the projection of the Euclidean gradient 
        # onto the tangent plane.
        # Euclidean grad for vertex i: -p / sin(R)
        # Tangent grad: ( -p/sin(R) - pts[i] * ( -p/sin(R) . pts[i] ) )
        
        grad = np.zeros_like(pts)
        
        # Vectorized update is hard because faces map to points.
        # Loop is fine for N=130.
        sin_radii = np.sin(radii)
        # Avoid division by zero
        sin_radii[sin_radii < _EPS] = _EPS
        
        for idx, face in enumerate(valid_faces):
            w = weights[idx]
            if w < 1e-6:
                continue
            
            p = normals[idx]
            R = radii[idx]
            s = sin_radii[idx]
            
            # Gradient for each vertex in the face
            for v_idx in face:
                # Euclidean gradient direction: -p / s
                eucl_grad = -p / s
                
                # Project to tangent space at pts[v_idx]
                # tangent_grad = eucl_grad - pts[v_idx] * (eucl_grad . pts[v_idx])
                proj = np.dot(eucl_grad, pts[v_idx])
                tangent_grad = eucl_grad - pts[v_idx] * proj
                
                grad[v_idx] += w * tangent_grad
        
        loss = np.log(sum_exp) * temp**2 # Scaling back roughly, or just use log(sum_exp)
        
        return loss, grad

    except Exception:
        return 0.0, np.zeros_like(pts)

def _gd_step(pts, grad, lr):
    """
    Perform a gradient descent step on the sphere.
    pts_new = normalize(pts - lr * grad)
    """
    new_pts = pts - lr * grad
    new_pts /= (np.linalg.norm(new_pts, axis=1, keepdims=True) + _EPS)
    return new_pts

def _local_polish_gd(pts, n_iters=100, lr_init=0.01):
    """
    Optimize using Gradient Descent on the soft-max surrogate.
    """
    best_pts = pts.copy()
    best_r = _compute_covering_radius(pts)
    
    lr = lr_init
    
    for i in range(n_iters):
        loss, grad = _soft_max_loss_and_grad(pts)
        
        if np.any(np.isnan(grad)):
            break
            
        pts = _gd_step(pts, grad, lr)
        
        # Check improvement
        curr_r = _compute_covering_radius(pts)
        
        if curr_r < best_r:
            best_r = curr_r
            best_pts = pts.copy()
            # If improved, maybe increase lr slightly? No, keep it stable.
        else:
            # If no improvement, reduce learning rate
            lr *= 0.9
            
    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    rng = np.random.default_rng(42)
    
    # Multi-restart strategy
    # 1. Fibonacci start
    # 2. Random starts
    
    best_pts = _initial_fib()
    best_r = _compute_covering_radius(best_pts)
    
    n_restarts = 10
    
    for restart in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
            
        if restart == 0:
            pts = _initial_fib()
        else:
            pts = _random_init(rng)
            
        # Phase 1: Coarse GD
        # Use a larger temperature initially to balance the whole sphere
        # Then refine with smaller temperature to minimize the max
        
        # We'll just run the standard _local_polish_gd which uses a fixed temp.
        # To improve, we can run it with different learning rates.
        
        # Time budget per restart
        phase_deadline = time.time() + 10.0
        
        iters = 0
        while time.time() < phase_deadline and iters < 50:
            pts = _local_polish_gd(pts, n_iters=10, lr_init=0.005 + 0.001 * restart)
            iters += 10
            
        curr_r = _compute_covering_radius(pts)
        if curr_r < best_r:
            best_r = curr_r
            best_pts = pts.copy()
            
    # Final polish on the best configuration with very fine steps
    if time.time() < deadline - 5.0:
        phase_deadline = time.time() + 5.0
        while time.time() < phase_deadline:
            # Use very small lr for final polishing
            best_pts = _local_polish_gd(best_pts, n_iters=5, lr_init=0.001)
            curr_r = _compute_covering_radius(best_pts)
            if curr_r < best_r:
                best_r = curr_r
            else:
                break
                
    # Final normalization
    best_pts = np.array(best_pts, dtype=float)
    best_pts /= (np.linalg.norm(best_pts, axis=1, keepdims=True) + _EPS)
    
    if not np.all(np.isfinite(best_pts)) or best_pts.shape != (N, 3):
        return _initial_fib()
        
    return best_pts
