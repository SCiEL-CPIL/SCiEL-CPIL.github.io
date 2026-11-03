"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import math

def _to_sphere(pts):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _compute_facet_circumradii(pts):
    """
    Compute angular circumradii for all facets of the convex hull.
    Returns an array of circumradii (in radians) and the hull simplices.
    Returns (np.array([np.pi]), None) if invalid.
    """
    pts = _to_sphere(pts)
    n = len(pts)
    
    # Vectorized coincident point check
    dists = np.linalg.norm(pts[:, np.newaxis] - pts[np.newaxis, :], axis=2)
    np.fill_diagonal(dists, np.inf)
    if np.min(dists) < 1e-8:
        return np.array([np.pi]), None
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.array([np.pi]), None
    
    simplices = hull.simplices
    if len(simplices) == 0:
        return np.array([np.pi]), None
        
    a = pts[simplices[:, 0]]
    b = pts[simplices[:, 1]]
    c = pts[simplices[:, 2]]
    
    # Normal to each facet
    normals = np.cross(b - a, c - a)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    
    valid = (norms[:, 0] > 1e-12)
    if not np.all(valid):
        return np.array([np.pi]), None
    
    normals = normals[valid] / norms[valid]
    a_v = a[valid]
    b_v = b[valid]
    c_v = c[valid]
    
    # For each facet, compute angular distance from normal to each vertex
    # The circumcenter is either +n or -n; pick the one with smaller max distance
    dot_a = np.clip(normals @ a_v.T, -1, 1)
    dot_b = np.clip(normals @ b_v.T, -1, 1)
    dot_c = np.clip(normals @ c_v.T, -1, 1)
    
    # Angular distances for +n
    da_pos = np.arccos(dot_a)
    db_pos = np.arccos(dot_b)
    dc_pos = np.arccos(dot_c)
    max_pos = np.maximum(np.maximum(da_pos, db_pos), dc_pos)
    
    # Angular distances for -n
    da_neg = np.arccos(-dot_a)
    db_neg = np.arccos(-dot_b)
    dc_neg = np.arccos(-dot_c)
    max_neg = np.maximum(np.maximum(da_neg, db_neg), dc_neg)
    
    # For each facet, the circumradius is min(max_pos, max_neg)
    # We also need to know which side (+ or -) was chosen for gradient calculation
    # is_pos = max_pos <= max_neg
    facet_radii = np.minimum(max_pos, max_neg)
    
    return facet_radii, simplices[valid]

def _covering_radius_fast(pts):
    """
    Fast covering radius computation via convex hull facet circumradii.
    Returns the covering radius in radians, or np.pi if invalid.
    """
    radii, _ = _compute_facet_circumradii(pts)
    if len(radii) == 0:
        return np.pi
    return np.max(radii)

def _smooth_covering_radius(pts, beta=50.0):
    """
    Smooth surrogate for the covering radius using log-sum-exp.
    This is differentiable and can be used with gradient-based optimization.
    Returns value and gradient.
    """
    pts = _to_sphere(pts)
    radii, simplices = _compute_facet_circumradii(pts)
    
    if len(radii) == 0 or simplices is None:
        return np.pi, np.zeros_like(pts)
    
    max_r = np.max(radii)
    if max_r < 1e-10:
        return 0.0, np.zeros_like(pts)
    
    shifted = radii - max_r
    log_sum = np.log(np.sum(np.exp(beta * shifted))) / beta
    smooth_val = max_r + log_sum
    
    # Gradient calculation
    # d/dx logsumexp(beta * r_i) = sum( exp(beta * r_i) / sum(exp(beta * r_i)) * beta * dr_i/dx )
    # We need dr_i/dx for each facet.
    # r_i is the max angular distance from the circumcenter pole to the vertices.
    # For a facet with vertices a, b, c and pole p (unit vector),
    # r = max(arccos(p.a), arccos(p.b), arccos(p.c)).
    # This is non-smooth. However, the log-sum-exp over facets smooths the max over facets.
    # The gradient of the smooth max over facets is a weighted sum of the gradients of the facet radii.
    
    # To get the gradient of a facet radius w.r.t points:
    # 1. Identify the "active" vertex (the one with max distance to pole).
    # 2. Gradient of arccos(p.v) w.r.t v is -p / sqrt(1 - (p.v)^2).
    # 3. Gradient w.r.t p is -v / sqrt(1 - (p.v)^2).
    # 4. p is the normalized cross product of (b-a) x (c-a).
    # This is complex. A simpler approximation often used:
    # The gradient of the covering radius is dominated by the "worst" facet.
    # We can approximate the gradient by moving points away from the circumcenter of the worst facets.
    
    # Let's use a finite difference approach for the gradient if analytic is too risky/complex,
    # OR use the analytic gradient of the smooth surrogate properly.
    
    # Actually, let's compute the gradient of the smooth value w.r.t the points.
    # smooth_val = logsumexp(beta * radii) / beta + max_r (approx)
    # grad_smooth_val = (1/sum_exp) * sum( exp(beta*r_i) * beta * grad_r_i )
    
    weights = np.exp(beta * shifted)
    weights /= np.sum(weights)
    
    grad = np.zeros_like(pts)
    
    # We need grad_r_i for each facet i.
    # r_i = max( d(p_i, v_{i,0}), d(p_i, v_{i,1}), d(p_i, v_{i,2}) )
    # Let k_i be the index of the vertex with max distance.
    # Then grad_{v_{i,k_i}} r_i = -p_i / sqrt(1 - (p_i . v_{i,k_i})^2)
    # And grad_{p_i} r_i = -v_{i,k_i} / sqrt(1 - (p_i . v_{i,k_i})^2)
    
    # We need p_i (poles) and the active vertex index.
    # We recompute poles and active vertices for the valid simplices.
    
    a = pts[simplices[:, 0]]
    b = pts[simplices[:, 1]]
    c = pts[simplices[:, 2]]
    
    normals = np.cross(b - a, c - a)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / norms
    
    dot_a = np.clip(normals @ a.T, -1, 1)
    dot_b = np.clip(normals @ b.T, -1, 1)
    dot_c = np.clip(normals @ c.T, -1, 1)
    
    da_pos = np.arccos(dot_a)
    db_pos = np.arccos(dot_b)
    dc_pos = np.arccos(dot_c)
    
    da_neg = np.arccos(-dot_a)
    db_neg = np.arccos(-dot_b)
    dc_neg = np.arccos(-dot_c)
    
    max_pos = np.maximum(np.maximum(da_pos, db_pos), dc_pos)
    max_neg = np.maximum(np.maximum(da_neg, db_neg), dc_neg)
    
    is_pos = max_pos <= max_neg
    
    # Select pole
    poles = np.where(is_pos[:, np.newaxis], normals, -normals)
    
    # Select active vertex index (0 for a, 1 for b, 2 for c)
    # For pos: max of da, db, dc
    # For neg: max of da_neg, db_neg, dc_neg
    
    # We need the gradient of the distance to the active vertex.
    # dist = arccos(p . v)
    # grad_v = -p / sqrt(1 - (p.v)^2)
    # grad_p = -v / sqrt(1 - (p.v)^2)
    
    # Let's compute the contribution of each facet to the gradient.
    # For facet i, let v_active be the vertex.
    # grad_r_i = grad_v * grad_p_term? No.
    # r_i depends on p_i and v_active.
    # p_i depends on a, b, c.
    # v_active is one of a, b, c.
    
    # This analytic gradient is quite involved. 
    # Given the time budget, a robust finite-difference gradient on the smooth surrogate 
    # might be safer and fast enough if we only do a few iterations, 
    # BUT L-BFGS needs many.
    
    # Alternative: Use the "repulsive" force heuristic which is effectively a gradient step.
    # Move each point away from the circumcenters of the facets that contain it, 
    # weighted by the facet's radius.
    
    # Let's stick to the finite difference for the gradient of the smooth surrogate 
    # but optimize it by only perturbing in the tangent space and using a small eps.
    # Actually, let's just use the analytic gradient of the *unsmoothed* max over the top few facets?
    # No, let's use a simple, robust gradient:
    # The gradient of the smooth max is a weighted sum of the gradients of the facet radii.
    # We approximate the gradient of a facet radius r_i w.r.t point x_j as:
    # - If x_j is a vertex of facet i:
    #   Let p_i be the pole. Let v_j be the point.
    #   Let d_j = arccos(p_i . v_j).
    #   If d_j is the max for this facet (active):
    #     grad_{v_j} r_i = -p_i / sqrt(1 - (p_i . v_j)^2)
    #     # Note: p_i also changes, but that's second order or handled by the chain rule.
    #     # For a first order approximation, we ignore the change in p_i w.r.t v_j.
    #     # This is a standard approximation in these problems.
    
    active_indices = np.zeros(len(simplices), dtype=int)
    # For pos facets
    pos_max_idx = np.argmax(np.column_stack([da_pos, db_pos, dc_pos]), axis=1)
    # For neg facets
    neg_max_idx = np.argmax(np.column_stack([da_neg, db_neg, dc_neg]), axis=1)
    
    active_indices = np.where(is_pos, pos_max_idx, neg_max_idx)
    
    # Get the active vertices and their dots with poles
    # We need to map active_indices to the actual points
    # For each facet i, active vertex index k in {0,1,2}
    # Point index in pts: simplices[i, k]
    
    active_point_indices = simplices[np.arange(len(simplices)), active_indices]
    active_points = pts[active_point_indices]
    
    # Compute dot product of pole and active point
    active_dots = np.einsum('ij,ij->i', poles, active_points)
    active_dots = np.clip(active_dots, -1, 1)
    
    denom = np.sqrt(1.0 - active_dots * active_dots)
    denom[denom < 1e-8] = 1e-8
    
    # Gradient of r_i w.r.t active_point is -pole / denom
    # This gradient is with respect to the point on the sphere.
    # We need to project it onto the tangent space? 
    # The smooth value is a function of the parameters (points). 
    # If we parameterize by points on the sphere, the gradient is in R^3.
    # We will project the step in the optimization.
    
    grad_vecs = -poles / denom[:, np.newaxis]
    
    # Accumulate gradients
    # weight_i = weights[i] * beta (from d logsumexp / dx = sum w_i * beta * r_i')
    # Actually, smooth_val = max_r + logsumexp(beta*(r_i - max_r))/beta
    # d smooth_val / dx = sum_i (exp(beta*(r_i-max_r)) / sum(exp(...))) * beta * dr_i/dx / beta
    # = sum_i w_i * dr_i/dx
    # where w_i = exp(...) / sum(...)
    
    weighted_grads = weights[:, np.newaxis] * grad_vecs
    
    for i in range(len(simplices)):
        idx = active_point_indices[i]
        grad[idx] += weighted_grads[i]
        
    return smooth_val, grad

def _fibonacci_spiral(n, offset=0.0):
    """Generate n points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + offset
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _local_optimize(pts, max_iter=100, step_size=0.02, deadline=None):
    """
    Local optimization with adaptive step size.
    """
    best_pts = pts.copy()
    best_val = _covering_radius_fast(best_pts)
    current_step = step_size
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Perturb a random point
        idx = np.random.randint(len(best_pts))
        direction = np.random.randn(3)
        direction = direction / np.linalg.norm(direction)
        
        new_pts = best_pts.copy()
        new_pts[idx] = best_pts[idx] + current_step * direction
        new_pts[idx] = new_pts[idx] / np.linalg.norm(new_pts[idx])
        
        new_val = _covering_radius_fast(new_pts)
        
        if new_val < best_val:
            best_pts = new_pts
            best_val = new_val
            current_step *= 1.01
        else:
            current_step *= 0.95
            
        if current_step < 1e-7:
            break
    
    return best_pts, best_val

def _gradient_refine(pts, max_iter=50, deadline=None):
    """
    Gradient-based refinement using the smooth surrogate.
    """
    current_pts = pts.copy()
    
    # Use L-BFGS-B on the flat vector
    def obj_and_grad(flat_pts):
        pts = _to_sphere(flat_pts.reshape(N, 3))
        val, grad = _smooth_covering_radius(pts, beta=80.0)
        # Project gradient to tangent space? 
        # L-BFGS-B works on the flat vector. 
        # The objective is invariant to scaling of individual points, 
        # so the gradient component along the radial direction is zero (or should be).
        # We can just pass the gradient.
        return val, grad.flatten()
        
    try:
        result = minimize(obj_and_grad, current_pts.flatten(), method='L-BFGS-B', jac=True,
                          options={'maxiter': max_iter, 'gtol': 1e-8, 'ftol': 1e-10})
        new_pts = _to_sphere(result.x.reshape(N, 3))
        new_val = _covering_radius_fast(new_pts)
        
        current_val = _covering_radius_fast(current_pts)
        if new_val < current_val:
            return new_pts, new_val
    except Exception:
        pass
        
    return current_pts, _covering_radius_fast(current_pts)

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    best_pts = None
    best_val = np.pi
    
    # Strategy 1: Fibonacci spiral
    pts = _fibonacci_spiral(N)
    val = _covering_radius_fast(pts)
    if val < best_val:
        best_val = val
        best_pts = pts.copy()
    
    # Local optimization from Fibonacci
    opt_pts, opt_val = _local_optimize(pts, max_iter=100, step_size=0.03, deadline=deadline)
    if opt_val < best_val:
        best_val = opt_val
        best_pts = opt_pts.copy()
    
    # Gradient refinement
    if time.time() < deadline - 20.0:
        grad_pts, grad_val = _gradient_refine(best_pts, max_iter=30, deadline=deadline)
        if grad_val < best_val:
            best_val = grad_val
            best_pts = grad_pts.copy()
            
    # Final local polish
    if time.time() < deadline - 5.0:
        final_pts, final_val = _local_optimize(best_pts, max_iter=50, step_size=0.01, deadline=deadline)
        if final_val < best_val:
            best_val = final_val
            best_pts = final_pts.copy()
    
    return _to_sphere(best_pts)
# EVOLVE-BLOCK-END
