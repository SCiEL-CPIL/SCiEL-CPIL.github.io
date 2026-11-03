"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms

def _project_to_tangent(vec, point):
    """Project vector to tangent space at point on unit sphere."""
    # vec is shape (3,) or (M, 3)
    # point is shape (3,) or (M, 3)
    dot = np.sum(vec * point, axis=-1, keepdims=True)
    return vec - dot * point

def _covering_radius_obj_grad(params, sample_points, lam=10.0):
    """
    Compute the Log-Sum-Exp approximation of the covering radius squared
    and its gradient w.r.t the flattened parameters.
    
    params: shape (3N,)
    sample_points: shape (M, 3)
    lam: temperature parameter for log-sum-exp
    """
    pts = params.reshape(N, 3)
    pts = _normalize(pts)
    
    M = len(sample_points)
    
    # Compute distances from each sample point to each config point
    # dist_sq[i, j] = |y_i - p_j|^2
    # y_i: (M, 1, 3), p_j: (1, N, 3)
    diff = sample_points[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=2)
    
    # Find nearest neighbor for each sample point
    min_idx = np.argmin(dist_sq, axis=1)
    min_dist_sq = dist_sq[np.arange(M), min_idx]
    
    # Log-Sum-Exp of min_dist_sq
    # L = (1/lam) * log( sum(exp(lam * min_dist_sq)) )
    # For numerical stability, subtract max
    max_val = np.max(min_dist_sq)
    exp_terms = np.exp(lam * (min_dist_sq - max_val))
    log_sum = np.log(np.sum(exp_terms))
    obj_val = (max_val + log_sum / lam)
    
    # Gradient
    # dL/dp_j = (1/lam) * d/dp_j [ log(sum_i exp(lam * d_i^2)) ]
    # = (1/lam) * ( sum_i exp(lam * d_i^2) * lam * d_i^2 * d/dp_j d_i^2 ) / sum_i exp(lam * d_i^2)
    # = sum_i w_i * d_i^2 * d/dp_j d_i^2
    # where w_i = exp(lam * (d_i^2 - max_val)) / sum_k exp(lam * (d_k^2 - max_val))
    
    # Note: d_i^2 depends on p_j only if j is the nearest neighbor?
    # Wait, the objective is max_y min_j d(y, p_j).
    # The Log-Sum-Exp is over the sample points y, of the value min_j d(y, p_j)^2.
    # Let f_i = min_j d(y_i, p_j)^2.
    # L = (1/lam) log sum_i exp(lam f_i).
    # dL/dp_j = (1/lam) * ( sum_i exp(lam f_i) * lam * df_i/dp_j ) / sum_i exp(lam f_i)
    # = sum_i w_i * df_i/dp_j
    # where w_i = exp(lam (f_i - max_f)) / sum_k exp(lam (f_k - max_f))
    
    # f_i = min_j d(y_i, p_j)^2.
    # df_i/dp_j is non-zero only if j is the argmin for sample i.
    # If j = argmin_i, then f_i = d(y_i, p_j)^2.
    # d/dp_j d(y_i, p_j)^2 = d/dp_j (2 - 2 y_i . p_j) = -2 y_i (projected to tangent space of p_j).
    # Actually, the gradient w.r.t the vector p_j in R^3 is -2 y_i.
    # But we must project to the tangent space of the sphere at p_j.
    # Grad_tangent = -2 * project(y_i, p_j) = -2 * (y_i - (y_i.p_j) p_j)
    
    # Calculate weights w_i
    exp_terms_norm = exp_terms / np.sum(exp_terms)
    
    # Initialize gradient
    grad_pts = np.zeros_like(pts)
    
    # For each sample i, find its nearest neighbor j
    # Accumulate gradient contribution
    # grad_pts[j] += w_i * (-2 * (y_i - (y_i.p_j) p_j))
    
    # Vectorized accumulation
    # min_idx gives j for each i
    # y_i is sample_points[i]
    # p_j is pts[min_idx[i]]
    
    # Compute projection term: y_i - (y_i . p_j) p_j
    # y_i: (M, 3)
    # p_j: (M, 3) (gathered)
    p_nearest = pts[min_idx]
    dots = np.sum(sample_points * p_nearest, axis=1, keepdims=True)
    proj_terms = sample_points - dots * p_nearest
    
    # Gradient contribution per sample: -2 * proj_terms * w_i
    weights = exp_terms_norm[:, np.newaxis]
    contribs = -2.0 * proj_terms * weights
    
    # Scatter add to grad_pts
    # grad_pts[j] += contribs[i]
    np.add.at(grad_pts, min_idx, contribs)
    
    # Flatten gradient
    grad_flat = grad_pts.flatten()
    
    return obj_val, grad_flat

def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    # 1. Initialize with Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Generate sample points for covering radius estimation
    # Use a dense grid or random points
    # We use random points for robustness, fixed seed for reproducibility
    np.random.seed(42)
    M = 5000
    sample_points = np.random.normal(0, 1, (M, 3))
    sample_points = _normalize(sample_points)
    
    # 3. Optimize
    # Use L-BFGS-B with the custom gradient
    x0_flat = x0.flatten()
    
    # Lambda controls the sharpness of the max approximation.
    # Higher lambda -> closer to true max, but noisier gradient.
    # Start with moderate lambda, maybe increase?
    # L-BFGS-B doesn't easily allow changing lambda during run without wrapper.
    # We can run multiple stages.
    
    # Stage 1: Lower lambda for smoothness
    lam1 = 5.0
    res1 = minimize(
        _covering_radius_obj_grad, 
        x0_flat, 
        args=(sample_points, lam1),
        method='L-BFGS-B', 
        jac=True,
        options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    # Stage 2: Higher lambda for precision
    lam2 = 20.0
    res2 = minimize(
        _covering_radius_obj_grad, 
        res1.x, 
        args=(sample_points, lam2),
        method='L-BFGS-B', 
        jac=True,
        options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    # Stage 3: Even higher lambda
    lam3 = 50.0
    res3 = minimize(
        _covering_radius_obj_grad, 
        res2.x, 
        args=(sample_points, lam3),
        method='L-BFGS-B', 
        jac=True,
        options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
    )
    
    points_final = _normalize(res3.x.reshape(N, 3))
    
    return points_final
# EVOLVE-BLOCK-END
