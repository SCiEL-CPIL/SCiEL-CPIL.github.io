# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def squares_in_square10() -> tuple:
    """
    Erdos "squares in a square" problem. Place n squares with pairwise disjoint
    interiors inside the unit square [0,1]^2 so as to MAXIMIZE the sum of their
    side lengths. Here n = 10.

    Returns:
        centers: np.ndarray of shape (10, 2) with the (x, y) center of each square.
        sides:   np.ndarray of shape (10,) with the side length of each square.
    The i-th axis-aligned square occupies
        [cx - s/2, cx + s/2] x [cy - s/2, cy + s/2].

    Strategy:
    1. Start with a known good packing (4-3-3 layout) with sum of sides 3.0.
    2. Use scipy.optimize.minimize with a penalty method to improve the packing.
    3. Return the best valid packing found.
    """
    n = 10
    tol = 1e-7
    
    # Initial layout: 4 squares in bottom row, 3 in middle, 3 in top.
    centers = np.zeros((n, 2), dtype=np.float64)
    sides = np.zeros(n, dtype=np.float64)
    
    # Row 1 (bottom): y_center = 0.125
    s1 = 0.25
    sides[0:4] = s1
    centers[0:4, 1] = s1 / 2
    centers[0:4, 0] = np.array([0.125, 0.375, 0.625, 0.875])
    
    # Row 2 (middle): y_center = 0.25 + (1/3)/2
    s2 = 1.0 / 3.0
    sides[4:7] = s2
    centers[4:7, 1] = 0.25 + s2 / 2
    centers[4:7, 0] = np.array([1.0/6.0, 0.5, 5.0/6.0])
    
    # Row 3 (top): y_center = 0.25 + 1/3 + (1/3)/2
    s3 = 1.0 / 3.0
    sides[7:10] = s3
    centers[7:10, 1] = 0.25 + s2 + s3 / 2
    centers[7:10, 0] = np.array([1.0/6.0, 0.5, 5.0/6.0])
    
    best_centers = centers.copy()
    best_sides = sides.copy()
    best_score = np.sum(sides)
    
    def is_valid(centers, sides):
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
            return False
        if np.any(sides <= 0):
            return False
        # Check inside unit square
        for i in range(n):
            cx, cy = centers[i]
            s = sides[i]
            if cx - s/2 < -tol or cx + s/2 > 1 + tol:
                return False
            if cy - s/2 < -tol or cy + s/2 > 1 + tol:
                return False
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                t = (sides[i] + sides[j]) / 2 - tol
                if abs(centers[i, 0] - centers[j, 0]) < t and abs(centers[i, 1] - centers[j, 1]) < t:
                    return False
        return True

    def objective(params):
        """
        Minimize negative sum of sides plus penalty for overlaps and boundary violations.
        params: [x1, y1, s1, x2, y2, s2, ..., x10, y10, s10]
        """
        centers = params[0:2*n:n].reshape(n, 1) # This is wrong, need to interleave properly
        # Let's reshape properly: params is [x1, y1, s1, x2, y2, s2, ...]
        # centers[:, 0] = params[0::3]
        # centers[:, 1] = params[1::3]
        # sides = params[2::3]
        
        cx = params[0::3]
        cy = params[1::3]
        s = params[2::3]
        
        # Objective: -sum(s)
        obj = -np.sum(s)
        
        # Boundary penalties
        # s > 0
        if np.any(s <= 0):
            obj += 1e6 * np.sum(np.maximum(0, -s))
            
        # Inside [0, 1]
        # cx - s/2 >= 0 => cx >= s/2
        # cx + s/2 <= 1 => cx <= 1 - s/2
        # cy - s/2 >= 0 => cy >= s/2
        # cy + s/2 <= 1 => cy <= 1 - s/2
        
        # Penalty for cx < s/2
        pen_cx_low = np.maximum(0, s/2 - cx)
        pen_cx_high = np.maximum(0, cx - (1 - s/2))
        pen_cy_low = np.maximum(0, s/2 - cy)
        pen_cy_high = np.maximum(0, cy - (1 - s/2))
        
        obj += 1000 * (np.sum(pen_cx_low) + np.sum(pen_cx_high) + np.sum(pen_cy_low) + np.sum(pen_cy_high))
        
        # Overlap penalties
        # For each pair i, j
        # dx = cx[i] - cx[j]
        # dy = cy[i] - cy[j]
        # sx = s[i] + s[j]
        # overlap_x = max(0, sx/2 - abs(dx))
        # overlap_y = max(0, sx/2 - abs(dy))
        # penalty = overlap_x * overlap_y
        
        total_overlap_pen = 0.0
        for i in range(n):
            for j in range(i+1, n):
                dx = cx[i] - cx[j]
                dy = cy[i] - cy[j]
                sx = s[i] + s[j]
                
                ox = sx/2 - abs(dx)
                oy = sx/2 - abs(dy)
                
                # Use a smooth approximation or just max(0, .) which is non-smooth but SLSQP can handle it sometimes, or use log-sum-exp for smoothness?
                # max(0, x) is 0 if x<0, x if x>0.
                # Let's use a soft version: 
                # soft_max(0, x) = log(1 + exp(x)) is always positive. 
                # We want 0 if no overlap.
                # If we use max(0, ox) * max(0, oy), it is 0 if either is 0.
                # This function is continuous but not differentiable at 0. SLSQP might struggle.
                # Let's try a quadratic penalty if overlapping:
                # If ox > 0 and oy > 0, penalty = (ox * oy)^2 ?
                # Or just ox * oy.
                
                if ox > 0 and oy > 0:
                    total_overlap_pen += ox * oy
        
        obj += 100 * total_overlap_pen
        
        return obj

    def to_flat(centers, sides):
        flat = np.zeros(3 * n)
        flat[0::3] = centers[:, 0]
        flat[1::3] = centers[:, 1]
        flat[2::3] = sides
        return flat

    def from_flat(flat):
        centers = np.zeros((n, 2))
        centers[:, 0] = flat[0::3]
        centers[:, 1] = flat[1::3]
        sides = flat[2::3]
        return centers, sides

    start_time = time.time()
    deadline = start_time + 580
    
    # Try a few random restarts/perturbations
    # We will run SLSQP from the initial point and a few perturbed points.
    
    x0 = to_flat(centers, sides)
    
    # Bounds:
    # s > 0, but we don't know upper bound. Let's say s <= 1.
    # cx, cy in [0, 1] roughly.
    bounds = []
    for i in range(n):
        bounds.append((1e-6, 1.0)) # cx
        bounds.append((1e-6, 1.0)) # cy
        bounds.append((1e-6, 1.0)) # s
        
    # Run optimization
    best_val = np.sum(sides)
    
    for restart in range(5):
        if time.time() > deadline:
            break
            
        if restart == 0:
            x_start = x0
        else:
            # Perturb initial solution
            x_start = x0 + np.random.normal(0, 0.05, size=3*n)
            # Ensure s is positive
            x_start[2::3] = np.maximum(x_start[2::3], 1e-3)
            # Clamp centers to [0,1]
            x_start[0::3] = np.clip(x_start[0::3], 0, 1)
            x_start[1::3] = np.clip(x_start[1::3], 0, 1)
            
        try:
            res = minimize(objective, x_start, method='SLSQP', bounds=bounds, 
                           options={'maxiter': 100, 'ftol': 1e-9})
            
            if res.success or res.fun < -best_val + 1e-6: # res.fun is -sum(s) + penalty
                # Check validity of the result
                c, s = from_flat(res.x)
                # Clean up: clamp to valid range
                s = np.maximum(s, 1e-6)
                c[:, 0] = np.clip(c[:, 0], s/2, 1 - s/2)
                c[:, 1] = np.clip(c[:, 1], s/2, 1 - s/2)
                
                if is_valid(c, s):
                    val = np.sum(s)
                    if val > best_val:
                        best_val = val
                        best_centers = c
                        best_sides = s
        except Exception as e:
            pass

    return best_centers, best_sides

# EVOLVE-BLOCK-END
