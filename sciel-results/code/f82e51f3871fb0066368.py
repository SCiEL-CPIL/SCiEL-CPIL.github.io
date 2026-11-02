"""
Heilbronn triangle problem: maximize minimum triangle area for 11 points.
Strategy:
1. Barycentric coordinate parameterization to enforce triangle constraint.
2. Use a smooth approximation of the minimum (log-sum-exp) to create a differentiable objective.
3. Perform multi-start optimization using L-BFGS-B on the smooth objective.
4. Refine the best solution using a more accurate (less smooth) objective or direct evaluation.
"""
import numpy as np
import time
import random
from scipy.optimize import minimize

N = 11
S3H = np.sqrt(3) / 2
DEADLINE = 140.0  # Safety margin before 150s limit

# Pre-compute triplet indices for vectorized area calculation
def _get_triplets(n):
    triplets = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                triplets.append((i, j, k))
    return np.array(triplets)

TRIPLETS = _get_triplets(N)
I_IDX = TRIPLETS[:, 0]
J_IDX = TRIPLETS[:, 1]
K_IDX = TRIPLETS[:, 2]

def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by three points."""
    n = len(pts)
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                dx1 = pts[j, 0] - pts[i, 0]
                dy1 = pts[j, 1] - pts[i, 1]
                dx2 = pts[k, 0] - pts[i, 0]
                dy2 = pts[k, 1] - pts[i, 1]
                cross = abs(dx1 * dy2 - dy1 * dx2)
                area = 0.5 * cross
                if area < min_area:
                    min_area = area
    return min_area

def _barycentric_to_cartesian(bary):
    """Convert barycentric coordinates (u, v, w) with u+v+w=1 to Cartesian (x, y).
    Triangle vertices: A(0,0), B(1,0), C(0.5, sqrt(3)/2).
    P = u*A + v*B + w*C = v*B + w*C
    x = v*1 + w*0.5 = v + 0.5*w
    y = v*0 + w*sqrt(3)/2 = w*sqrt(3)/2
    """
    # bary: shape (N, 3)
    u, v, w = bary[:, 0], bary[:, 1], bary[:, 2]
    x = v + 0.5 * w
    y = S3H * w
    return np.column_stack((x, y))

def _cartesian_to_barycentric(pts):
    """Convert Cartesian (x, y) to barycentric coordinates (u, v, w).
    x = v + 0.5*w
    y = S3H*w  => w = y/S3H
    v = x - 0.5*w
    u = 1 - v - w
    """
    x, y = pts[:, 0], pts[:, 1]
    w = y / S3H
    v = x - 0.5 * w
    u = 1 - v - w
    return np.column_stack((u, v, w))

def _soft_min_objective(x_flat, b):
    """
    Objective function for minimization.
    x_flat: flattened barycentric coordinates (N*3).
    b: temperature parameter for log-sum-exp approximation of min.
    Returns: -soft_min_area + penalty.
    """
    bary = x_flat.reshape(N, 3)
    
    # Soft normalization to ensure barycentric coords are positive and sum to 1
    # Actually, L-BFGS-B handles bounds. We can use bounds (0,1) for each coord.
    # But we need them to sum to 1. This is a simplex constraint.
    # Standard trick: parameterize with N*2 free parameters (drop w=1-u-v) and use bounds u>=0, v>=0, u+v<=1.
    # Let's switch to that for better constraint handling.
    # x_flat will be (N, 2) containing (u, v). w = 1 - u - v.
    
    uv = x_flat.reshape(N, 2)
    u, v = uv[:, 0], uv[:, 1]
    w = 1.0 - u - v
    
    # Check validity (u,v,w >= 0)
    # If any w < 0, it's outside. Penalty.
    # But L-BFGS-B with bounds u>=0, v>=0, u+v<=1 is tricky directly.
    # Easier: use bounds u in [0,1], v in [0,1] and add penalty for u+v>1.
    # Or better: transform to ensure valid barycentrics?
    # Let's stick to the penalty method for simplicity and robustness within L-BFGS-B.
    
    # Calculate penalty for invalid barycentric coords
    penalty = 0.0
    if np.any(w < -1e-9):
        penalty += 1000.0 * np.sum(np.clip(-w, 0, None)**2)
    if np.any(u < -1e-9):
        penalty += 1000.0 * np.sum(np.clip(-u, 0, None)**2)
    if np.any(v < -1e-9):
        penalty += 1000.0 * np.sum(np.clip(-v, 0, None)**2)
        
    # Clip to 0 for calculation to avoid nan in log/exp? No, we want penalty to push it back.
    # But for area calc, we can just use the values.
    
    pts = _barycentric_to_cartesian(np.column_stack((u, v, w)))
    
    # Compute areas vectorized
    p_i = pts[I_IDX]
    p_j = pts[J_IDX]
    p_k = pts[K_IDX]
    
    dx1 = p_j[:, 0] - p_i[:, 0]
    dy1 = p_j[:, 1] - p_i[:, 1]
    dx2 = p_k[:, 0] - p_i[:, 0]
    dy2 = p_k[:, 1] - p_i[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    # Log-sum-exp approximation of min
    # min(a) ~ (1/b) * log(sum(exp(-b*a)))
    # We want to MAXIMIZE min, so MINIMIZE -min.
    # -min(a) ~ -(1/b) * log(sum(exp(-b*a)))
    
    # Numerical stability: subtract max
    shifted = -b * areas
    max_shifted = np.max(shifted)
    log_sum_exp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    soft_min_val = (1.0 / b) * log_sum_exp
    
    obj = -soft_min_val + penalty
    return obj

def _random_barycentric(rng):
    """Generate random valid barycentric coordinates."""
    bary = np.zeros((N, 3))
    for i in range(N):
        # Generate random point in simplex
        r1 = rng.random()
        r2 = rng.random()
        if r1 + r2 > 1:
            r1 = 1 - r1
            r2 = 1 - r2
        bary[i] = [1 - r1 - r2, r1, r2]
    return bary

def construct():
    start_time = time.time()
    rng = random.Random(42)
    
    best_score = 0.0
    best_pts = None
    
    # Multi-start optimization
    n_starts = 50
    # Allocate time per start
    time_per_start = (DEADLINE - (time.time() - start_time)) / n_starts
    
    for s in range(n_starts):
        if time.time() - start_time > DEADLINE - 5:
            break
            
        # Initialize random barycentric coords
        bary_init = _random_barycentric(rng)
        x0 = bary_init[:, :2].flatten()  # Use u, v
        
        # Define bounds: u in [0, 1], v in [0, 1]
        bounds = [(0, 1)] * (2 * N)
        
        # Objective function with fixed b parameter
        # Start with larger b (less smooth, more accurate) or smaller b (smoother)?
        # Smaller b is smoother, easier for optimizer to navigate.
        # Larger b is closer to true min.
        # Let's use b=20.0
        b = 20.0
        
        def obj(x):
            return _soft_min_objective(x, b)
        
        try:
            # L-BFGS-B optimization
            result = minimize(
                obj,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50, 'maxfun': 100}
            )
            
            x_opt = result.x
            bary_opt = np.zeros((N, 3))
            bary_opt[:, 0] = x_opt[0::2]
            bary_opt[:, 1] = x_opt[1::2]
            bary_opt[:, 2] = 1 - bary_opt[:, 0] - bary_opt[:, 1]
            
            # Project to valid barycentrics if needed (clip)
            bary_opt = np.clip(bary_opt, 0, None)
            # Normalize? No, if clipped, sum might not be 1.
            # If w was negative, we clipped to 0. u+v might be > 1.
            # Let's just check if it's valid.
            if np.any(bary_opt < -1e-9) or np.any(np.sum(bary_opt, axis=1) > 1 + 1e-9):
                # Invalid, skip or project
                # Simple projection: clip and renormalize?
                # For now, skip if invalid.
                continue
            
            pts = _barycentric_to_cartesian(bary_opt)
            
            # Evaluate true min area
            score = _min_triangle_area(pts)
            
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
                
        except Exception as e:
            pass

    # If no valid point found, fallback
    if best_pts is None:
        # Generate a simple valid configuration
        bary = _random_barycentric(rng)
        best_pts = _barycentric_to_cartesian(bary)
        
    return best_pts
