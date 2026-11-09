import numpy as np
import time
from scipy.optimize import minimize

def _is_valid(centers, sides, tol=1e-7):
    n = len(sides)
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if cx - s/2 < -tol or cx + s/2 > 1 + tol:
            return False
        if cy - s/2 < -tol or cy + s/2 > 1 + tol:
            return False
    for i in range(n):
        for j in range(i+1, n):
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            t = (sides[i] + sides[j]) / 2.0 - tol
            if dx < t and dy < t:
                return False
    return True

def _objective(params, n=10, tol=1e-7):
    centers = params[:2*n].reshape(n, 2)
    sides = params[2*n:]
    
    pen = 0.0
    # Boundary penalties
    for i in range(n):
        cx, cy = centers[i]
        s = sides[i]
        if s <= 0:
            pen += 1e6 * (1.0 - s)
            continue
        if cx - s/2 < -tol:
            pen += 1e6 * (-tol - (cx - s/2))
        if cx + s/2 > 1 + tol:
            pen += 1e6 * ((cx + s/2) - (1 + tol))
        if cy - s/2 < -tol:
            pen += 1e6 * (-tol - (cy - s/2))
        if cy + s/2 > 1 + tol:
            pen += 1e6 * ((cy + s/2) - (1 + tol))
            
    # Overlap penalties
    for i in range(n):
        for j in range(i+1, n):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            t = (sides[i] + sides[j]) / 2.0 - tol
            vx = t - abs(dx)
            vy = t - abs(dy)
            sx = np.maximum(0, vx)
            sy = np.maximum(0, vy)
            pen += 1e5 * (sx * sy)
            
    return -np.sum(sides) + pen

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 580
    
    best_centers = None
    best_sides = None
    best_score = -np.inf
    
    def update_best(centers, sides):
        nonlocal best_centers, best_sides, best_score
        if _is_valid(centers, sides):
            score = np.sum(sides)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_sides = sides.copy()

    # Construct the specific 3.0 configuration
    # 4 Large (0.375) at corners
    # 6 Small (0.25) in center strip and side gaps
    
    s_large = 0.375
    s_small = 0.25
    
    centers = np.zeros((n, 2))
    sides = np.zeros(n)
    
    # 4 Large
    # 0: BL
    centers[0] = [s_large/2, s_large/2]
    sides[0] = s_large
    # 1: BR
    centers[1] = [1 - s_large/2, s_large/2]
    sides[1] = s_large
    # 2: TL
    centers[2] = [s_large/2, 1 - s_large/2]
    sides[2] = s_large
    # 3: TR
    centers[3] = [1 - s_large/2, 1 - s_large/2]
    sides[3] = s_large
    
    # 4 Small in central strip (x: 0.375 to 0.625, width 0.25)
    # y positions: 0.125, 0.375, 0.625, 0.875
    centers[4] = [0.5, 0.125]
    sides[4] = s_small
    centers[5] = [0.5, 0.375]
    sides[5] = s_small
    centers[6] = [0.5, 0.625]
    sides[6] = s_small
    centers[7] = [0.5, 0.875]
    sides[7] = s_small
    
    # 2 Small in side gaps
    # Left gap: x [0, 0.375], y [0.375, 0.625]
    # Center x = 0.1875, Center y = 0.5
    centers[8] = [0.1875, 0.5]
    sides[8] = s_small
    
    # Right gap: x [0.625, 1.0], y [0.375, 0.625]
    # Center x = 0.8125, Center y = 0.5
    centers[9] = [0.8125, 0.5]
    sides[9] = s_small
    
    update_best(centers, sides)
    
    # Try to optimize from this starting point
    # We use a penalty-based objective to keep it simple and robust
    # If optimization fails or doesn't improve, we keep the static solution
    
    if time.time() < deadline - 10:
        params_init = np.concatenate([centers.flatten(), sides])
        
        try:
            # Use Powell method which doesn't require gradients and handles bounds well if we add them, 
            # but here we rely on penalties. 
            # We can also try L-BFGS-B with bounds if we reformulate, but penalty is safer for non-convex overlaps.
            res = minimize(_objective, params_init, args=(n,), method='Powell',
                           options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
            params_opt = res.x
            c_opt = params_opt[:2*n].reshape(n, 2)
            s_opt = np.maximum(params_opt[2*n:], 1e-6)
            update_best(c_opt, s_opt)
        except Exception:
            pass

    if best_centers is None:
        # Fallback to the static configuration if something went wrong
        best_centers = centers
        best_sides = sides

    return best_centers, best_sides
