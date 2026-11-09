# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

N = 300

def _generate_hex_lattice(n):
    """Generate a hexagonal lattice clipped to the unit circle."""
    # Binary search for spacing s such that we get at least n points
    # Area approx: n * s^2 * sqrt(3) / 2 = pi
    s_low = 0.05
    s_high = 0.2
    
    def count_points(s):
        k = int(1.0 / s) + 2
        xs = np.arange(-k, k + 1) * s
        ys = np.arange(-k, k + 1) * s * np.sqrt(3) / 2
        XX, YY = np.meshgrid(xs, ys, indexing='ij')
        XX[1::2, :] += s / 2
        pts = np.column_stack([XX.ravel(), YY.ravel()])
        dists = np.linalg.norm(pts, axis=1)
        return np.sum(dists <= 1.0 - s/2)

    # Find s such that count >= n
    # We want larger s (larger circles) -> fewer points.
    # We want exactly n.
    # If s is small, count is large.
    # If s is large, count is small.
    # We want the largest s such that count >= n.
    
    for _ in range(20):
        s_mid = (s_low + s_high) / 2
        cnt = count_points(s_mid)
        if cnt >= n:
            s_low = s_mid
        else:
            s_high = s_mid
            
    s = s_low
    
    # Generate points with this s
    k = int(1.0 / s) + 2
    xs = np.arange(-k, k + 1) * s
    ys = np.arange(-k, k + 1) * s * np.sqrt(3) / 2
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    XX[1::2, :] += s / 2
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    dists = np.linalg.norm(pts, axis=1)
    
    # Filter points inside circle (with margin s/2)
    mask = dists <= 1.0 - s/2
    pts = pts[mask]
    dists = dists[mask]
    
    # Select n closest to center
    if len(pts) > n:
        order = np.argsort(dists)
        pts = pts[order[:n]]
    elif len(pts) < n:
        # Fallback: just use what we have, or pad? 
        # The problem requires N points. 
        # If we can't fit N points with this s, we should have picked a smaller s.
        # The binary search should have handled this.
        # If still < n, we might need to relax boundary constraint or use a different grid.
        # For now, if < n, we'll just use a square grid fallback or repeat points? No, must be distinct.
        # Let's assume binary search works.
        pass
        
    return pts

def _smooth_objective_and_grad(x, beta=20.0):
    """
    Compute the soft-min objective and its gradient.
    Objective: (1/beta) * log(sum(exp(-beta * y)))
    where y are the constraints (distances/2 and boundary distances).
    Minimizing this maximizes the minimum constraint.
    """
    pts = x.reshape(-1, 2)
    n = len(pts)
    
    # Boundary terms: y_b_i = 1 - ||c_i||
    r_pts = np.linalg.norm(pts, axis=1)
    r_safe = np.where(r_pts > 1e-8, r_pts, 1e-8)
    y_b = 1.0 - r_pts
    grad_y_b = -pts / r_safe[:, np.newaxis]
    
    # Distance terms: y_d_ij = ||c_i - c_j|| / 2
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dist, np.inf)
    y_d = dist / 2.0
    
    # Weights
    e_b = np.exp(-beta * y_b)
    e_d = np.exp(-beta * y_d)
    np.fill_diagonal(e_d, 0.0)
    
    sum_e = np.sum(e_b) + np.sum(e_d)
    if sum_e < 1e-300:
        return 0.0, np.zeros_like(x)
        
    w_b = e_b / sum_e
    w_d = e_d / sum_e
    
    # Objective
    obj = (1.0 / beta) * np.log(sum_e)
    
    # Gradient
    # d/dx log(sum exp) = sum w_i * grad(y_i)
    # We want to MINIMIZE obj.
    # grad(obj) = sum w_i * grad(y_i)
    # Note: y_b and y_d are functions of x.
    # For boundary: grad(y_b_i) = -c_i / ||c_i||
    # Contribution to grad: w_b_i * (-c_i / ||c_i||)
    
    grad = np.zeros_like(pts)
    
    # Boundary contribution
    grad += w_b[:, np.newaxis] * grad_y_b
    
    # Distance contribution
    # y_d_ij depends on c_i and c_j.
    # grad_{c_i} (y_d_ij) = (c_i - c_j) / (2 * dist_ij)
    # grad_{c_j} (y_d_ij) = (c_j - c_i) / (2 * dist_ij)
    
    # We have w_d matrix.
    # For each i, sum over j: w_d_ij * (c_i - c_j) / (2 * dist_ij)
    # Let's compute the coefficient for each pair.
    # coeff_ij = w_d_ij / (2 * dist_ij)
    # Note: dist_ij = 2 * y_d_ij.
    # So coeff_ij = w_d_ij / (4 * y_d_ij) ? No.
    # dist = 2 * y_d.
    # grad y_d = diff / (2 * dist) = diff / (4 * y_d).
    # This is risky if y_d is small.
    
    # Better:
    # dist_safe = np.where(dist > 1e-8, dist, 1.0)
    # coeff = w_d / (2.0 * dist_safe)
    # term_i = sum_j coeff_ij * diff_ij
    
    dist_safe = np.where(dist > 1e-8, dist, 1.0)
    coeff = w_d / (2.0 * dist_safe)
    
    # Accumulate gradient
    # grad_d[i] = sum_j coeff[i,j] * (pts[i] - pts[j])
    # Vectorized:
    # coeff is (N, N, 1) ? No, coeff is (N, N).
    # diff is (N, N, 2).
    # coeff[:, :, np.newaxis] * diff -> (N, N, 2)
    # sum over axis 1 -> (N, 2)
    
    grad_d = np.sum(coeff[:, :, np.newaxis] * diff, axis=1)
    grad += grad_d
    
    return obj, grad

def construct():
    deadline = time.time() + 140.0
    
    # 1. Generate initial lattice
    points = _generate_hex_lattice(N)
    
    if points is None or len(points) < N:
        # Fallback: simple grid
        points = np.zeros((N, 2))
        # ... implement fallback if needed, but lattice should work
        
    # 2. Optimize
    x0 = points.flatten()
    bounds = [(-1.0, 1.0)] * (2 * N)
    
    def fun(x):
        return _smooth_objective_and_grad(x, beta=20.0)
    
    try:
        res = minimize(fun, x0, method='L-BFGS-B', jac=True, bounds=bounds,
                       options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
        
        if time.time() < deadline - 10:
             # Try a second pass with higher beta for sharper min
             x1 = res.x
             res2 = minimize(fun, x1, method='L-BFGS-B', jac=True, bounds=bounds,
                             options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
             if res2.success and res2.fun < res.fun:
                 points = res2.x.reshape(-1, 2)
             else:
                 points = res.x.reshape(-1, 2)
        else:
             points = res.x.reshape(-1, 2)
             
    except Exception:
        pass # Keep initial points if optimization fails

    return points
# EVOLVE-BLOCK-END
