import numpy as np
from scipy.optimize import minimize
import time

def squares_in_square10() -> tuple:
    n = 10
    start_time = time.time()
    deadline = start_time + 590  # 10s safety margin
    
    best_score = -np.inf
    best_centers = None
    best_sides = None
    
    # Objective function
    def objective(x):
        # x: [cx0, cy0, s0, cx1, cy1, s1, ...]
        cx = x[0::3]
        cy = x[1::3]
        s = x[2::3]
        
        # Boundary penalty
        pen = 0.0
        # s must be > 0
        pen += np.sum(np.maximum(0, 1e-6 - s))
        
        # Bounds: [0, 1]
        pen += np.sum(np.maximum(0, s/2 - cx))
        pen += np.sum(np.maximum(0, cx + s/2 - 1))
        pen += np.sum(np.maximum(0, s/2 - cy))
        pen += np.sum(np.maximum(0, cy + s/2 - 1))
        
        # Overlap penalty
        # Vectorized pairwise differences
        # cx: (10,), cy: (10,), s: (10,)
        # dx: (10, 10)
        cx_diff = np.abs(cx[:, None] - cx[None, :])
        cy_diff = np.abs(cy[:, None] - cy[None, :])
        s_sum_half = (s[:, None] + s[None, :]) / 2.0
        
        # Overlap in x: s_sum_half - cx_diff (if positive)
        # Overlap in y: s_sum_half - cy_diff (if positive)
        ov_x = np.maximum(0, s_sum_half - cx_diff)
        ov_y = np.maximum(0, s_sum_half - cy_diff)
        
        # Intersection area
        inter_area = ov_x * ov_y
        # Sum over all pairs (i != j)
        # Diagonal is 0 because s_i/2 - 0 = s_i/2 > 0? No, cx_diff[i,i]=0, s_sum_half[i,i]=s_i.
        # ov_x[i,i] = s_i. ov_y[i,i] = s_i. inter_area[i,i] = s_i^2.
        # We must exclude diagonal.
        inter_area[np.diag_indices(n)] = 0
        pen += np.sum(inter_area)
        
        # We want to maximize sum(s), so minimize -sum(s) + penalty
        # Squaring penalty makes it very steep
        return pen**2 - np.sum(s)

    def run_optimization(x0, max_iter=2000):
        try:
            res = minimize(objective, x0, method='Nelder-Mead', 
                           options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8})
            return res.x, res.fun
        except Exception:
            return x0, objective(x0)

    # Generate starting points
    starts = []
    
    # 1. Baseline 4x4 grid
    k = 4
    side = 1.0 / k
    idx = np.arange(n)
    cols = idx % k
    rows = idx // k
    cx0 = (cols + 0.5) * side
    cy0 = (rows + 0.5) * side
    s0 = np.full(n, side)
    x0 = np.empty(n*3)
    x0[0::3] = cx0
    x0[1::3] = cy0
    x0[2::3] = s0
    starts.append(x0)
    
    # 2. 3x3 grid + 1 small (shrink 3x3 to make space)
    # 9 squares of side 0.32, 1 square of side 0.1?
    # Let's just randomize around 0.3 size
    np.random.seed(42)
    for i in range(50):
        cx = np.random.uniform(0.2, 0.8, n)
        cy = np.random.uniform(0.2, 0.8, n)
        s = np.random.uniform(0.2, 0.35, n)
        x = np.empty(n*3)
        x[0::3] = cx
        x[1::3] = cy
        x[2::3] = s
        starts.append(x)
        
    # 3. Specific heuristic: 1 big square in corner, others around
    # Big square 0.5, 0.5, 0.5
    # Others smaller
    cx = [0.25, 0.75, 0.75, 0.75, 0.25, 0.25, 0.75, 0.25, 0.5, 0.5]
    cy = [0.25, 0.25, 0.75, 0.5, 0.75, 0.5, 0.5, 0.25, 0.25, 0.75]
    s = [0.5, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    x = np.empty(n*3)
    x[0::3] = cx
    x[1::3] = cy
    x[2::3] = s
    starts.append(x)
    
    # 4. "Strip" packing: 2 columns of 5?
    # 5 squares of 0.5 x 0.2? No, squares.
    # 5 squares of 0.2 in col 1, 5 squares of 0.2 in col 2?
    # Try 10 squares of 0.316 (1/sqrt(10))
    # Random starts cover this.

    # Optimize each start
    for x0 in starts:
        if time.time() > deadline - 5:
            break
            
        x_res, f_res = run_optimization(x0)
        
        # Extract results
        cx = x_res[0::3]
        cy = x_res[1::3]
        s = x_res[2::3]
        
        # Validate
        if np.any(s <= 0) or np.any(~np.isfinite(cx)) or np.any(~np.isfinite(cy)) or np.any(~np.isfinite(s)):
            continue
            
        centers = np.column_stack((cx, cy))
        sides = s
        
        # Check bounds with tol
        tol = 1e-7
        if np.any(centers[:, 0] - sides/2 < -tol) or np.any(centers[:, 0] + sides/2 > 1 + tol) or \
           np.any(centers[:, 1] - sides/2 < -tol) or np.any(centers[:, 1] + sides/2 > 1 + tol):
            continue
            
        # Check overlaps
        valid = True
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j])/2 - tol
                if abs(centers[i,0] - centers[j,0]) < t and abs(centers[i,1] - centers[j,1]) < t:
                    valid = False
                    break
            if not valid:
                break
                
        if valid:
            score = np.sum(sides)
            if score > best_score:
                best_score = score
                best_centers = centers.copy()
                best_sides = sides.copy()
                
    # If no valid solution found, return baseline (which is valid)
    if best_centers is None:
        k = 4
        side = 1.0 / k
        idx = np.arange(n)
        cols = idx % k
        rows = idx // k
        cx = (cols + 0.5) * side
        cy = (rows + 0.5) * side
        centers = np.stack([cx, cy], axis=1).astype(np.float64)
        sides = np.full(n, side, dtype=np.float64)
        return centers, sides

    return best_centers, best_sides
