import numpy as np
from scipy.optimize import minimize
import time

N = 11
SQRT3 = np.sqrt(3)
TRI_AREA = 0.5 * 1.0 * (SQRT3 / 2.0)

def compute_areas_vectorized(points):
    """
    Compute areas of all triangles formed by N points.
    Returns array of shape (N, N, N) where A[i,j,k] is area of triangle i,j,k.
    """
    n = points.shape[0]
    # p1, p2, p3
    # Area = 0.5 * | (p2-p1) x (p3-p1) |
    # In 2D, cross product is scalar: (x2-x1)*(y3-y1) - (y2-y1)*(x3-x1)
    
    # Expand dimensions to broadcast
    # p_i: (n, 2)
    # We want A[i, j, k] = 0.5 * | (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i) |
    
    x = points[:, 0]
    y = points[:, 1]
    
    # dx_ij = x_j - x_i -> shape (n, n)
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    
    # We need to combine dx[i,j] with dy[i,k] and dy[i,j] with dx[i,k]
    # Let's construct the cross product term C[i, j, k]
    # C[i, j, k] = dx[i, j] * dy[i, k] - dy[i, j] * dx[i, k]
    
    # dx[i, j] is shape (n, n). We want to multiply with dy[i, k].
    # dy[i, k] is shape (n, n).
    # We need C[i, j, k].
    
    # Term 1: dx[i, j] * dy[i, k]
    # dx is (n, n) -> (n, n, 1)
    # dy is (n, n) -> (n, 1, n)
    # But indices: dx[i,j] and dy[i,k].
    # dx[:, j, None] * dy[:, None, k]? No.
    
    # Let's do it explicitly for clarity and safety, then optimize if needed.
    # Actually, for N=11, 11^3 = 1331 operations is trivial.
    
    areas = np.zeros((n, n, n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                # 2 * Area = | (x_j - x_i)*(y_k - y_i) - (y_j - y_i)*(x_k - x_i) |
                cross = (points[j, 0] - points[i, 0]) * (points[k, 1] - points[i, 1]) - \
                        (points[j, 1] - points[i, 1]) * (points[k, 0] - points[i, 0])
                areas[i, j, k] = 0.5 * abs(cross)
    return areas

def penalty_and_grad(points):
    """
    Calculate penalty for points outside the triangle and its gradient.
    Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    Inequalities:
    1. y >= 0
    2. sqrt(3)*x - y <= sqrt(3)*(1-x) => sqrt(3)*x - y + sqrt(3)*x - sqrt(3) <= 0? 
       Line from (1,0) to (0.5, sqrt(3)/2):
       Slope = (sqrt(3)/2 - 0) / (0.5 - 1) = -sqrt(3).
       Eq: y - 0 = -sqrt(3) (x - 1) => y = -sqrt(3)x + sqrt(3) => sqrt(3)x + y - sqrt(3) <= 0.
    3. Line from (0,0) to (0.5, sqrt(3)/2):
       Slope = sqrt(3).
       Eq: y = sqrt(3)x => y - sqrt(3)x <= 0.
       
    So constraints for inside:
    g1 = -y <= 0  -> y >= 0
    g2 = sqrt(3)x + y - sqrt(3) <= 0
    g3 = y - sqrt(3)x <= 0
    
    Penalty = sum(max(0, g_i)^2)
    """
    x = points[:, 0]
    y = points[:, 1]
    
    g1 = -y
    g2 = SQRT3 * x + y - SQRT3
    g3 = y - SQRT3 * x
    
    # Penalty
    pen = np.sum(np.maximum(0, g1)**2) + np.sum(np.maximum(0, g2)**2) + np.sum(np.maximum(0, g3)**2)
    
    # Gradients
    # d(pen)/d(points)
    grad_pen = np.zeros_like(points)
    
    # g1 = -y. d(g1)/dy = -1. If g1 > 0, d(pen)/dy = 2*g1*(-1) = -2*g1.
    mask1 = g1 > 0
    if np.any(mask1):
        grad_pen[mask1, 1] += -2 * g1[mask1]
        
    # g2 = SQRT3*x + y - SQRT3. d/dx = SQRT3, d/dy = 1.
    # If g2 > 0, d(pen)/dx = 2*g2*SQRT3, d(pen)/dy = 2*g2.
    mask2 = g2 > 0
    if np.any(mask2):
        grad_pen[mask2, 0] += 2 * g2[mask2] * SQRT3
        grad_pen[mask2, 1] += 2 * g2[mask2]
        
    # g3 = y - SQRT3*x. d/dx = -SQRT3, d/dy = 1.
    # If g3 > 0, d(pen)/dx = 2*g3*(-SQRT3), d(pen)/dy = 2*g3.
    mask3 = g3 > 0
    if np.any(mask3):
        grad_pen[mask3, 0] += -2 * g3[mask3] * SQRT3
        grad_pen[mask3, 1] += 2 * g3[mask3]
        
    return pen, grad_pen

def smooth_min_area_and_grad(points, temp=1.0):
    """
    Approximate min area using log-sum-exp (softmin) for differentiability.
    Min A ~ -1/temp * log(sum(exp(-A_i/temp)))
    """
    areas = compute_areas_vectorized(points)
    # Flatten and remove zeros/degenerate triangles? 
    # Triangles with i=j=k or i=j etc have area 0. 
    # We only care about distinct i,j,k.
    mask = np.ones(areas.shape, dtype=bool)
    # Zero out diagonals and duplicates
    for i in range(N):
        for j in range(N):
            if i == j:
                mask[i, j, :] = False
            mask[i, j, i] = False
            mask[i, j, j] = False
            mask[i, j, k] if False else None # Just simple loops for mask creation is safer
            
    # Let's just extract unique triangles i < j < k
    areas_flat = []
    indices = []
    for i in range(N):
        for j in range(i+1, N):
            for k in range(j+1, N):
                areas_flat.append(areas[i, j, k])
                indices.append((i, j, k))
    areas_flat = np.array(areas_flat)
    
    if len(areas_flat) == 0:
        return 0.0, np.zeros_like(points)
        
    # Softmin
    # To avoid overflow, subtract min
    min_a = np.min(areas_flat)
    exp_terms = np.exp(-(areas_flat - min_a) / temp)
    sum_exp = np.sum(exp_terms)
    log_sum = np.log(sum_exp)
    soft_min_val = min_a - temp * log_sum
    
    # Gradient of soft_min_val w.r.t points
    # d(soft_min_val)/d(A_m) = -exp_m / sum_exp
    # d(A_m)/d(points) needs to be computed.
    
    weights = exp_terms / sum_exp
    
    grad_points = np.zeros_like(points)
    
    for idx, (i, j, k) in enumerate(indices):
        w = weights[idx]
        if w < 1e-10:
            continue
            
        # Area A = 0.5 * | Cross |
        # Let Cross = (x_j - x_i)(y_k - y_i) - (y_j - y_i)(x_k - x_i)
        # A = 0.5 * |Cross|
        # dA/dx_l = 0.5 * sign(Cross) * dCross/dx_l
        
        cross = (points[j, 0] - points[i, 0]) * (points[k, 1] - points[i, 1]) - \
                (points[j, 1] - points[i, 1]) * (points[k, 0] - points[i, 0])
        
        if abs(cross) < 1e-12:
            # Degenerate, gradient is undefined/zero, skip
            continue
            
        sign_cross = np.sign(cross)
        
        # dCross/dx_i
        # Cross = x_j y_k - x_j y_i - x_i y_k + x_i y_i - y_j x_k + y_j x_i + y_i x_k - y_i x_i
        # Wait, easier:
        # Cross = x_j(y_k - y_i) - y_j(x_k - x_i)
        # d/dx_i: -y_j
        # d/dy_i: -x_j
        # d/dx_j: y_k - y_i
        # d/dy_j: -(x_k - x_i)
        # d/dx_k: -y_j
        # d/dy_k: x_j
        
        d_cross_dx_i = -points[j, 1]
        d_cross_dy_i = -points[j, 0]
        
        d_cross_dx_j = points[k, 1] - points[i, 1]
        d_cross_dy_j = -(points[k, 0] - points[i, 0])
        
        d_cross_dx_k = -points[j, 1]
        d_cross_dy_k = points[j, 0]
        
        factor = 0.5 * sign_cross * w
        
        grad_points[i, 0] += factor * d_cross_dx_i
        grad_points[i, 1] += factor * d_cross_dy_i
        grad_points[j, 0] += factor * d_cross_dx_j
        grad_points[j, 1] += factor * d_cross_dy_j
        grad_points[k, 0] += factor * d_cross_dx_k
        grad_points[k, 1] += factor * d_cross_dy_k
        
    return soft_min_val, grad_points

def construct():
    start_time = time.time()
    time_limit = 110.0
    
    best_points = None
    best_score = -np.inf
    
    # Initializations
    inits = []
    
    # 1. Uniform random
    for seed in range(10):
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
        
    # 2. Grid-like (Vogel or similar)
    # Use a pre-computed good layout? Or just dense grid.
    # Let's try a few structured inits.
    # Center point + 10 on boundary?
    # Just add a few more randoms with different seeds
    for seed in range(10, 30):
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

    for init_pts in inits:
        if time.time() - start_time > time_limit - 5:
            break
            
        # Custom Gradient Descent
        current_pts = init_pts.copy()
        lr = 0.01
        temp = 0.05
        
        # Adaptive LR schedule
        for step in range(1000):
            if time.time() - start_time > time_limit - 5:
                break
                
            # Compute Objective and Gradient
            min_area, grad_area = smooth_min_area_and_grad(current_pts, temp)
            pen, grad_pen = penalty_and_grad(current_pts)
            
            # Objective: -min_area + lambda * pen
            # We want to maximize min_area, so minimize -min_area.
            # Penalty weight should be high initially, then decay?
            # Or just keep it constant high enough to stay inside.
            lam = 100.0
            
            grad_obj = -grad_area + lam * grad_pen
            
            # Gradient Descent Step
            # Normalize gradient to prevent huge steps
            norm_g = np.linalg.norm(grad_obj)
            if norm_g > 1e-8:
                grad_obj = grad_obj / norm_g
                
            current_pts -= lr * grad_obj
            
            # Project to triangle (simple clamp/projection)
            # If penalty is high, project hard
            x = current_pts[:, 0]
            y = current_pts[:, 1]
            
            # Simple projection:
            # 1. y < 0 -> y = 0
            # 2. y > sqrt(3)*x -> y = sqrt(3)*x
            # 3. y > sqrt(3)*(1-x) -> y = sqrt(3)*(1-x)
            
            # This is approximate but helps stability
            mask_y_low = y < 0
            if np.any(mask_y_low):
                current_pts[mask_y_low, 1] = 0
                
            mask_y_high_left = y > SQRT3 * x
            if np.any(mask_y_high_left):
                current_pts[mask_y_high_left, 1] = SQRT3 * current_pts[mask_y_high_left, 0]
                
            mask_y_high_right = y > SQRT3 * (1 - x)
            if np.any(mask_y_high_right):
                current_pts[mask_y_high_right, 1] = SQRT3 * (1 - current_pts[mask_y_high_right, 0])
                
            # Decay temperature
            if step % 50 == 0:
                temp *= 0.9
                lr *= 0.95
                
        # Evaluate final points
        # Use exact min area for scoring
        areas = compute_areas_vectorized(current_pts)
        # Extract valid triangles
        valid_areas = []
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    valid_areas.append(areas[i, j, k])
        min_area_exact = np.min(valid_areas)
        
        pen_final, _ = penalty_and_grad(current_pts)
        
        if pen_final < 1e-5:
            score = min_area_exact
        else:
            score = min_area_exact - 100 * pen_final
            
        if score > best_score:
            best_score = score
            best_points = current_pts.copy()
            
    # Polish the best point found with L-BFGS-B
    if best_points is not None and time.time() - start_time < time_limit - 10:
        try:
            def polish_obj(flat):
                pts = flat.reshape(N, 2)
                areas = compute_areas_vectorized(pts)
                valid_areas = []
                for i in range(N):
                    for j in range(i+1, N):
                        for k in range(j+1, N):
                            valid_areas.append(areas[i, j, k])
                min_a = np.min(valid_areas)
                p, _ = penalty_and_grad(pts)
                return -min_a + 1000 * p
                
            res = minimize(polish_obj, best_points.flatten(), method='L-BFGS-B', 
                           options={'maxiter': 500, 'maxfev': 5000})
            
            if res.success or res.fun < -best_score + 1000*penalty_and_grad(best_points)[0]:
                new_pts = res.x.reshape(N, 2)
                areas = compute_areas_vectorized(new_pts)
                valid_areas = []
                for i in range(N):
                    for j in range(i+1, N):
                        for k in range(j+1, N):
                            valid_areas.append(areas[i, j, k])
                min_a_new = np.min(valid_areas)
                pen_new, _ = penalty_and_grad(new_pts)
                
                if pen_new < 1e-5:
                    score_new = min_a_new
                else:
                    score_new = min_a_new - 1000 * pen_new
                    
                if score_new > best_score:
                    best_score = score_new
                    best_points = new_pts
        except Exception:
            pass

    if best_points is None:
        # Fallback
        return np.array([[0.5, SQRT3/6]] * N)
        
    return best_points
