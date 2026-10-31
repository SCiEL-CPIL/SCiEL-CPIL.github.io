import numpy as np
from scipy.optimize import minimize
import time

N = 11
SQRT3 = np.sqrt(3)
H = SQRT3 / 2.0

def get_indices():
    """Precompute indices for all unique triplets."""
    i_idx, j_idx, k_idx = np.indices((N, N, N))
    mask = (i_idx < j_idx) & (j_idx < k_idx)
    i_flat = i_idx[mask]
    j_flat = j_idx[mask]
    k_flat = k_idx[mask]
    return i_flat, j_flat, k_flat

I_IDX, J_IDX, K_IDX = get_indices()
NUM_TRI = len(I_IDX)

def compute_areas(points):
    """
    Compute signed areas (2 * area) for all triangles.
    Returns array of shape (NUM_TRI,)
    """
    p1 = points[I_IDX]
    p2 = points[J_IDX]
    p3 = points[K_IDX]
    
    # 2 * Area = (x2-x1)(y3-y1) - (y2-y1)(x3-x1)
    # = x2y3 - x2y1 - x1y3 + x1y1 - y2x3 + y2x1 + y1x3 - y1x1
    # = x2(y3-y1) + x1(y1-y3) + x3(y1-y2)
    areas2 = p1[:, 0] * (p2[:, 1] - p3[:, 1]) + \
             p2[:, 0] * (p3[:, 1] - p1[:, 1]) + \
             p3[:, 0] * (p1[:, 1] - p2[:, 1])
    return areas2

def penalty_boundary(points):
    """
    Penalty for points outside the triangle.
    Triangle: (0,0), (1,0), (0.5, H)
    Lines:
    1. y >= 0
    2. x >= 0
    3. y <= -SQRT3 * (x - 1) => SQRT3 * x + y <= SQRT3
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # d1: distance to y=0 (below)
    d1 = np.maximum(0, -y)
    # d2: distance to x=0 (left)
    d2 = np.maximum(0, -x)
    # d3: distance to line SQRT3*x + y = SQRT3 (right/above)
    # Normal is (SQRT3, 1). Distance is (SQRT3*x + y - SQRT3) / 2 (since norm is 2)
    # We just penalize the violation of the linear inequality SQRT3*x + y <= SQRT3
    d3 = np.maximum(0, SQRT3 * x + y - SQRT3)
    
    return np.sum(d1**2) + np.sum(d2**2) + np.sum(d3**2)

def objective_and_grad(params):
    """
    Objective: Minimize -alpha + lambda * sum(max(0, alpha - |A_ijk|/2)^2) + boundary_penalty
    Note: The task maximizes the smallest area. Let alpha be the target min area.
    We want max alpha such that Area_ijk >= alpha for all ijk.
    Equivalent to min -alpha subject to Area_ijk >= alpha.
    Penalty method: min -alpha + lambda * sum( max(0, alpha - Area_ijk)^2 )
    
    params: [x1, y1, ..., x11, y11, alpha]
    """
    points = params[:2*N].reshape(N, 2)
    alpha = params[2*N]
    
    areas2 = compute_areas(points)
    areas = np.abs(areas2) * 0.5
    
    # Violations: alpha - area
    viol = alpha - areas
    active = viol > 0
    penalty_term = np.sum(viol[active]**2)
    
    b_pen = penalty_boundary(points)
    
    # We need to tune lambda. Let's start with 1.0 and increase if needed, or keep fixed.
    # A fixed lambda might not push alpha high enough. 
    # Actually, if we minimize -alpha + lambda * penalty, 
    # if penalty is 0, we can increase alpha until it hits a constraint.
    # The gradient of -alpha is -1. The gradient of penalty w.r.t alpha is 2*sum(viol).
    # At optimum, -1 + 2*lambda*sum(viol) = 0 => sum(viol) = 1/(2*lambda).
    # This means active constraints will have small positive violations if lambda is small.
    # We want violations to be 0. So lambda should be large?
    # Or we can just maximize alpha directly by treating it as a variable and using a large penalty.
    
    obj = -alpha + 100.0 * penalty_term + 1000.0 * b_pen
    
    # Gradients
    grad = np.zeros_like(params)
    
    # Gradient w.r.t alpha
    grad[2*N] = -1.0 + 200.0 * np.sum(viol[active])
    
    # Gradient w.r.t points
    # d/dp_i sum (alpha - |A_k|)^2 for k where i is involved
    # Let's compute gradient of areas w.r.t points first.
    # This is tricky with abs. 
    # d|A|/dp = sign(A) * dA/dp
    # dA_k/dp_i = ...
    
    # We can compute the gradient of the penalty term numerically or analytically.
    # Analytical is faster.
    # Let's accumulate gradients for each point.
    
    # We need d(areas)/d(points)
    # area_k = 0.5 * | det(p_j - p_i, p_k - p_i) |
    # Let's use the formula:
    # 2 * area = x1(y2-y3) + x2(y3-y1) + x3(y1-y2)
    # d(2*area)/dx1 = y2 - y3
    # d(2*area)/dy1 = -(x2 - x3) = x3 - x2
    # Similarly for x2, y2, x3, y3.
    
    # Since we have vectorized areas2, we can compute the Jacobian of areas2 w.r.t points.
    # But it's sparse.
    # For each triangle k (i, j, m):
    # dA2_k/dx_i = y_j - y_m
    # dA2_k/dy_i = x_m - x_j
    # dA2_k/dx_j = y_m - y_i
    # dA2_k/dy_j = x_i - x_m
    # dA2_k/dx_m = y_i - y_j
    # dA2_k/dy_m = x_j - x_i
    
    # We only care about active triangles (where alpha > area).
    # Note: area = 0.5 * |A2|.
    # d(area)/dp = 0.5 * sign(A2) * d(A2)/dp
    
    # Let's gather the active triangles
    active_indices = np.where(active)[0]
    
    if len(active_indices) > 0:
        i_act = I_IDX[active_indices]
        j_act = J_IDX[active_indices]
        m_act = K_IDX[active_indices]
        
        a2_act = areas2[active_indices]
        signs = np.sign(a2_act)
        # If a2 is 0, sign is 0, gradient is 0 (subgradient, but 0 is safe)
        
        viol_act = viol[active_indices]
        
        # Factor: 2 * lambda * viol * 0.5 * sign = lambda * viol * sign
        # lambda = 100.0
        factors = 100.0 * viol_act * signs
        
        # We need to add factors * dA2/dp to the gradient of the penalty part of the objective.
        # The objective is -alpha + 100 * sum(viol^2).
        # dObj/dp = 100 * sum 2*viol * d(viol)/dp = 200 * sum viol * d(-area)/dp = -200 * sum viol * d(area)/dp
        # d(area)/dp = 0.5 * sign(A2) * dA2/dp
        # So dObj/dp = -100 * sum viol * sign(A2) * dA2/dp
        # factors above is 100 * viol * sign.
        # So we subtract factors * dA2/dp from the gradient?
        # Wait, dObj/dp = -100 * sum (viol * sign * dA2/dp).
        # So we accumulate -factors * dA2/dp into grad.
        
        # Let's use np.add.at for accumulation
        
        # dA2/dx_i = y_j - y_m
        dy_jm = points[j_act, 1] - points[m_act, 1]
        dx_mi = points[m_act, 0] - points[j_act, 0]
        
        # dA2/dx_j = y_m - y_i
        dy_mi = points[m_act, 1] - points[i_act, 1]
        dx_im = points[i_act, 0] - points[m_act, 0]
        
        # dA2/dx_m = y_i - y_j
        dy_ij = points[i_act, 1] - points[j_act, 1]
        dx_ji = points[j_act, 0] - points[i_act, 0]
        
        # Accumulate for point i
        # grad_x_i -= factors * dy_jm
        # grad_y_i -= factors * dx_mi
        
        np.add.at(grad, i_act*2, -factors * dy_jm)
        np.add.at(grad, i_act*2 + 1, -factors * dx_mi)
        
        np.add.at(grad, j_act*2, -factors * dy_mi)
        np.add.at(grad, j_act*2 + 1, -factors * dx_im)
        
        np.add.at(grad, m_act*2, -factors * dy_ij)
        np.add.at(grad, m_act*2 + 1, -factors * dx_ji)
        
        # Boundary penalty gradient
        x = points[:, 0]
        y = points[:, 1]
        b_pen_val = penalty_boundary(points)
        
        # d(b_pen)/dx
        # d1: max(0, -y) -> 0
        # d2: max(0, -x) -> -1 if x<0
        # d3: max(0, SQRT3*x + y - SQRT3) -> SQRT3 if violated
        
        gx = np.zeros(N)
        gy = np.zeros(N)
        
        mask_x = x < 0
        gx[mask_x] = -2.0 * (-x[mask_x]) # d/dx (-x)^2 = -2(-x) = 2x? No.
        # Penalty is sum(max(0, -x)^2). If x<0, val = x^2. d/dx = 2x.
        # Wait, max(0, -x) is -x if x<0. Square is x^2. Deriv is 2x.
        # So gx += 2 * x * (x<0)
        gx += 2.0 * x * mask_x
        
        mask_y = y < 0
        gy += 2.0 * y * mask_y
        
        mask_3 = (SQRT3 * x + y - SQRT3) > 0
        # val = (SQRT3*x + y - SQRT3)^2
        # d/dx = 2 * (SQRT3*x + y - SQRT3) * SQRT3
        # d/dy = 2 * (SQRT3*x + y - SQRT3) * 1
        v3 = SQRT3 * x + y - SQRT3
        gx += 2.0 * v3 * SQRT3 * mask_3
        gy += 2.0 * v3 * mask_3
        
        grad[:N*2:2] += 1000.0 * gx
        grad[1:N*2:2] += 1000.0 * gy

    return obj, grad

def construct():
    start_time = time.time()
    time_limit = 115.0
    
    best_points = None
    best_score = -np.inf
    
    # Initial guesses
    # 1. Centroid + random
    # 2. Symmetric grid
    
    inits = []
    
    # Random inits
    for seed in range(50):
        if time.time() - start_time > time_limit - 20:
            break
        rng = np.random.RandomState(seed)
        pts = np.zeros((N, 2))
        for i in range(N):
            while True:
                u = rng.rand()
                v = rng.rand()
                if u + v <= 1:
                    break
            w = 1 - u - v
            pts[i, 0] = v * 1.0 + w * 0.5
            pts[i, 1] = w * SQRT3 / 2.0
        inits.append(pts)
        
    # One symmetric init
    # 11 points:
    # 1 center (0.5, H/3)
    # 3 near vertices
    # 3 mid edges
    # 3 inner ring?
    sym_pts = np.array([
        [0.5, H/3],
        [0.1, H/6], [0.9, H/6], [0.5, 5*H/6],
        [0.5, H/6], [0.25, H/2], [0.75, H/2],
        [0.25, 3*H/4], [0.75, 3*H/4],
        [0.5, 2*H/3],
        [0.5, H/12]
    ])
    # Ensure inside
    inits.append(sym_pts)
    
    for init_pts in inits:
        if time.time() - start_time > time_limit - 5:
            break
            
        # Initial alpha: estimate from min area of init
        init_areas = np.abs(compute_areas(init_pts)) * 0.5
        init_alpha = np.min(init_areas) * 0.9
        
        init_params = np.concatenate([init_pts.flatten(), [init_alpha]])
        
        try:
            res = minimize(
                objective_and_grad,
                init_params,
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            opt_params = res.x
            opt_points = opt_params[:2*N].reshape(N, 2)
            opt_alpha = opt_params[2*N]
            
            # Verify constraints
            pen = penalty_boundary(opt_points)
            if pen > 1e-4:
                continue
                
            areas = np.abs(compute_areas(opt_points)) * 0.5
            min_area = np.min(areas)
            
            # The score is the min area (normalized? The task says "divides by its area" in the notebook, 
            # but the interface returns points. The evaluator measures "raw value". 
            # The current program returns points. The evaluator calculates the score.
            # So we just need to return the points that maximize the min area.
            
            if min_area > best_score:
                best_score = min_area
                best_points = opt_points.copy()
                
        except Exception as e:
            pass

    if best_points is None:
        # Fallback
        return np.array([[0.5, H/3]] * N)
        
    return best_points
