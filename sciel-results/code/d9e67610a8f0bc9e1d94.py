import numpy as np
from scipy.optimize import minimize

def _compute_score(points):
    points = np.array(points)
    n = len(points)
    if n < 2:
        return 0.0
    
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-10:
                return -np.inf
            dists.append(d)
    
    diameter = max(dists)
    if diameter < 1e-10:
        return -np.inf
    
    scale = 2.0 / diameter
    log_product = np.sum(np.log(dists)) + (n * (n - 1) / 2.0) * np.log(scale)
    return log_product

def _objective(points_flat):
    n = 8
    points = np.array(points_flat).reshape(n, 2)
    
    # Compute pairwise distances
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(points[i] - points[j])
            if d < 1e-10:
                return 1e10 # Penalty for collision
            dists.append(d)
            
    dists = np.array(dists)
    
    # Sum of log distances
    sum_log_dists = np.sum(np.log(dists))
    
    # Soft max of distances to approximate diameter
    # Using power mean with high exponent
    p = 20
    soft_max_diam = np.sum(dists**p)**(1.0/p)
    
    # The term to subtract is C * log(D_max)
    # C = n*(n-1)/2
    C = n * (n - 1) / 2.0
    
    # We want to maximize sum_log_dists - C * log(D_max)
    # So minimize negative of that
    val = sum_log_dists - C * np.log(soft_max_diam)
    
    return -val

def _gradient(points_flat):
    n = 8
    points = np.array(points_flat).reshape(n, 2)
    grad = np.zeros_like(points)
    
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dists[i, j] = np.linalg.norm(points[i] - points[j])
                
    # Gradient of sum log dists
    # d/dx_k sum_{i<j} log d_ij
    # For a specific pair (i,j), grad_i = (p_i - p_j) / d_ij^2, grad_j = (p_j - p_i) / d_ij^2
    for i in range(n):
        for j in range(i + 1, n):
            d2 = dists[i, j]**2
            if d2 < 1e-10:
                continue
            diff = points[i] - points[j]
            grad[i] += diff / d2
            grad[j] -= diff / d2
            
    # Gradient of soft max term
    # f = C * log( (sum d^p)^{1/p} ) = C/p * log(sum d^p)
    # df/dx = C/p * (1/sum d^p) * sum (p d^{p-1} * d d/dx)
    # df/dx = C * (1/sum d^p) * sum (d^{p-1} * grad d)
    
    p = 20
    C = n * (n - 1) / 2.0
    
    sum_dp = np.sum(dists[dists > 0]**p) # Avoid 0^p issues, though dists > 0
    # Actually dists is symmetric, we used upper triangle for log sum, but soft max should use all unique pairs
    # Let's recompute sum_dp using unique pairs to be consistent with C
    sum_dp_unique = 0.0
    weighted_grad = np.zeros_like(points)
    
    for i in range(n):
        for j in range(i + 1, n):
            d = dists[i, j]
            if d < 1e-10:
                continue
            dp = d**p
            sum_dp_unique += dp
            # grad d_ij wrt p_i is (p_i - p_j)/d
            grad_d_i = (points[i] - points[j]) / d
            grad_d_j = (points[j] - points[i]) / d
            
            # Term: d^{p-1} * grad d
            weight = d**(p-1)
            weighted_grad[i] += weight * grad_d_i
            weighted_grad[j] += weight * grad_d_j
            
    if sum_dp_unique > 0:
        coeff = C / sum_dp_unique
        grad -= coeff * weighted_grad
        
    return grad

def construct():
    n = 8
    best_score = -np.inf
    best_points = None
    
    # Initial candidates
    candidates = []
    
    # 1. Regular Octagon
    angles = [2 * np.pi * k / 8 for k in range(8)]
    reg_points = np.array([[np.cos(a), np.sin(a)] for a in angles])
    candidates.append(reg_points)
    
    # 2. Two Concentric Squares
    # Inner square radius r1, outer r2
    # Try a few ratios
    for r_ratio in [0.5, 0.7, 0.8, 0.9]:
        r1 = r_ratio
        r2 = 1.0
        pts = []
        for k in range(4):
            a1 = 2 * np.pi * k / 4 + np.pi/8 # Rotate inner to avoid alignment
            a2 = 2 * np.pi * k / 4
            pts.append([r1 * np.cos(a1), r1 * np.sin(a1)])
            pts.append([r2 * np.cos(a2), r2 * np.sin(a2)])
        candidates.append(np.array(pts))
        
    # 3. Square with 4 points on diagonals?
    # Just try a few random symmetric perturbations of the octagon
    
    for init_pts in candidates:
        x0 = init_pts.flatten()
        
        # Optimize
        try:
            result = minimize(_objective, x0, method='L-BFGS-B', jac=_gradient, 
                              options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-12})
            
            if result.success:
                curr_points = result.x.reshape(n, 2)
                curr_score = _compute_score(curr_points)
                
                if curr_score > best_score:
                    best_score = curr_score
                    best_points = curr_points
                    
        except Exception:
            pass

    # If no valid points found, fallback to regular octagon
    if best_points is None:
        best_points = reg_points

    # Final check and return
    pts = best_points
    # Ensure distinctness
    valid = True
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(pts[i] - pts[j]) < 1e-5:
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        return [[float(np.cos(2 * np.pi * k / 8)), float(np.sin(2 * np.pi * k / 8))] for k in range(8)]
        
    return [[float(p[0]), float(p[1])] for p in pts]
