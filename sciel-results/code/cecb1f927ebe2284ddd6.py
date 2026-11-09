import numpy as np
import time
from scipy.optimize import minimize

def _is_valid_vectorized(centers, sides, tol=1e-7):
    """
    Vectorized validity check.
    centers: (N, 2)
    sides: (N,)
    """
    n = len(sides)
    if centers.shape != (n, 2) or sides.shape != (n,):
        return False
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(sides)):
        return False
    if np.any(sides <= 0):
        return False
        
    # Boundary check
    # Square i is inside if cx - s/2 >= -tol and cx + s/2 <= 1 + tol
    # cx - s/2 >= -tol  => cx >= s/2 - tol
    # cx + s/2 <= 1 + tol => cx <= 1 - s/2 + tol
    if np.any(centers[:, 0] - sides/2 < -tol): return False
    if np.any(centers[:, 0] + sides/2 > 1 + tol): return False
    if np.any(centers[:, 1] - sides/2 < -tol): return False
    if np.any(centers[:, 1] + sides/2 > 1 + tol): return False
    
    # Overlap check
    # Overlap if |cx_i - cx_j| < (s_i + s_j)/2 - tol AND |cy_i - cy_j| < (s_i + s_j)/2 - tol
    dx = np.abs(centers[:, 0][:, np.newaxis] - centers[:, 0][np.newaxis, :])
    dy = np.abs(centers[:, 1][:, np.newaxis] - centers[:, 1][np.newaxis, :])
    sep = (sides[:, np.newaxis] + sides[np.newaxis, :]) / 2.0 - tol
    mask = ~np.eye(n, dtype=bool)
    
    # Forbidden if dx < sep AND dy < sep
    if np.any((dx < sep) & (dy < sep) & mask):
        return False
        
    return True

def _initial_packing_grid():
    """
    Returns the known good packing with sum 3.0.
    2 squares of 0.5, 8 squares of 0.25.
    """
    centers = np.zeros((10, 2))
    sides = np.zeros(10)
    
    # 2 Large at bottom
    centers[0] = [0.25, 0.25]
    sides[0] = 0.5
    centers[1] = [0.75, 0.25]
    sides[1] = 0.5
    
    # 8 Small in top half (y in [0.5, 1.0])
    # Two rows of 4
    for k in range(8):
        row = k // 4  # 0 or 1
        col = k % 4   # 0 to 3
        cy = 0.625 + row * 0.25
        cx = 0.125 + col * 0.25
        centers[2+k] = [cx, cy]
        sides[2+k] = 0.25
        
    return centers, sides

def _penalty_objective(vars, lambda_boundary=100.0, lambda_overlap=1000.0):
    """
    vars: array of length 30 (10 centers x 2, 10 sides)
    Returns negative of (sum(sides) - penalties) so we minimize it.
    """
    n = 10
    centers = vars[:20].reshape(10, 2)
    sides = vars[20:]
    
    # Penalty for boundary violations
    # We want centers[i, 0] in [sides[i]/2, 1 - sides[i]/2]
    # Violation is distance outside this range.
    # Since we use L-BFGS-B with bounds, we might not need boundary penalty if we set bounds correctly.
    # However, sides are variable, so bounds for centers depend on sides.
    # L-BFGS-B bounds must be static. So we cannot easily constrain centers based on sides using bounds alone.
    # We must use penalty for boundaries.
    
    # Boundary penalty:
    # cx < s/2 -> penalty (s/2 - cx)^2
    # cx > 1 - s/2 -> penalty (cx - (1 - s/2))^2
    pen_b = 0.0
    for i in range(n):
        s = sides[i]
        cx = centers[i, 0]
        cy = centers[i, 1]
        
        lo_x = s / 2.0
        hi_x = 1.0 - s / 2.0
        if cx < lo_x:
            pen_b += (lo_x - cx) ** 2
        elif cx > hi_x:
            pen_b += (cx - hi_x) ** 2
            
        lo_y = s / 2.0
        hi_y = 1.0 - s / 2.0
        if cy < lo_y:
            pen_b += (lo_y - cy) ** 2
        elif cy > hi_y:
            pen_b += (cy - hi_y) ** 2
            
    # Penalty for overlaps
    # Overlap in x: max(0, (s_i + s_j)/2 - |cx_i - cx_j|)
    # Overlap in y: max(0, (s_i + s_j)/2 - |cy_i - cy_j|)
    # Product of overlaps is penalized.
    pen_o = 0.0
    for i in range(n):
        for j in range(i+1, n):
            s_i = sides[i]
            s_j = sides[j]
            sep = (s_i + s_j) / 2.0
            
            dx = abs(centers[i, 0] - centers[j, 0])
            dy = abs(centers[i, 1] - centers[j, 1])
            
            ox = max(0.0, sep - dx)
            oy = max(0.0, sep - dy)
            
            if ox > 0 and oy > 0:
                pen_o += ox * oy
    
    total_penalty = lambda_boundary * pen_b + lambda_overlap * pen_o
    objective_value = np.sum(sides) - total_penalty
    
    # We minimize negative objective
    return -objective_value

def _polish_solution(centers, sides, deadline):
    """
    Run L-BFGS-B to maximize sum(sides) subject to soft constraints.
    """
    n = 10
    vars0 = np.concatenate([centers.flatten(), sides])
    
    # Bounds for L-BFGS-B
    # Centers: [0, 1] (soft constraint handles tighter bounds)
    # Sides: [0.01, 1.0] (must be > 0)
    bounds = []
    for i in range(n):
        bounds.append((0.0, 1.0)) # cx
        bounds.append((0.0, 1.0)) # cy
    for i in range(n):
        bounds.append((0.01, 1.0)) # side
    
    # We will run multiple optimizations with different lambda values to balance
    # maximizing size vs enforcing validity.
    
    best_val = np.sum(sides)
    best_centers = centers.copy()
    best_sides = sides.copy()
    
    lambdas = [10.0, 50.0, 100.0, 500.0, 1000.0, 5000.0]
    
    for lam_b in lambdas:
        for lam_o in lambdas:
            if time.time() > deadline:
                break
            
            # Reset to previous best for this lambda step? Or continue from last?
            # Continuing from last is usually better for annealing-like effect.
            # But if we get stuck, resetting to grid might help.
            # Let's try continuing.
            
            res = minimize(_penalty_objective, vars0, method='L-BFGS-B', bounds=bounds, 
                           args=(lam_b, lam_o), options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success or res.fun < -best_val:
                new_vars = res.x
                new_centers = new_vars[:20].reshape(10, 2)
                new_sides = new_vars[20:]
                
                # Check if this new solution is valid with strict tolerance
                # If it is, it's a candidate for best.
                # Even if not strictly valid, it might be a better starting point for higher lambda.
                
                # We want to keep track of the best VALID solution.
                # But the optimizer might find a high-sum solution that is slightly invalid.
                # We will check validity at the end.
                
                # For the purpose of chaining optimizations, we just update vars0.
                vars0 = new_vars
                
                # If this specific configuration is valid, update best
                if _is_valid_vectorized(new_centers, new_sides, tol=1e-7):
                    val = np.sum(new_sides)
                    if val > best_val:
                        best_val = val
                        best_centers = new_centers.copy()
                        best_sides = new_sides.copy()
    
    return best_centers, best_sides, best_val

def squares_in_square10() -> tuple:
    n = 10
    tol = 1e-7
    start_time = time.time()
    deadline = start_time + 620 # Safety margin
    
    # Best solution found so far
    best_centers, best_sides = _initial_packing_grid()
    best_val = np.sum(best_sides)
    
    rng = np.random.default_rng(42)
    
    # List of starting points
    starts = []
    
    # 1. The standard grid
    c, s = _initial_packing_grid()
    starts.append((c, s))
    
    # 2. Perturbed grid
    for _ in range(5):
        c, s = _initial_packing_grid()
        c += rng.uniform(-0.1, 0.1, c.shape)
        s += rng.uniform(-0.05, 0.05, s.shape)
        s = np.clip(s, 0.05, 0.9)
        # Clip centers to be roughly inside
        for i in range(n):
            si = s[i]
            c[i, 0] = np.clip(c[i, 0], 0.1, 0.9)
            c[i, 1] = np.clip(c[i, 1], 0.1, 0.9)
        starts.append((c, s))
        
    # 3. Try some random configurations with larger squares
    # E.g. 1 large, 9 small
    for _ in range(5):
        c = np.random.rand(n, 2) * 0.8 + 0.1
        s = np.zeros(n)
        s[0] = rng.uniform(0.5, 0.8)
        s[1:] = rng.uniform(0.2, 0.4, n-1)
        # Place large square in a corner
        c[0] = [s[0]/2, s[0]/2]
        starts.append((c, s))
        
    # 4. Try a 3x3 grid of small squares + 1 large?
    # 9 squares of 1/3? Sum = 3.0.
    # 1 square of 0.5, 9 squares of 0.25? Sum = 0.5 + 2.25 = 2.75. Worse.
    
    # Run optimization on each start
    for c, s in starts:
        if time.time() > deadline:
            break
            
        # Ensure start is not completely crazy
        if not np.all(np.isfinite(c)) or not np.all(np.isfinite(s)):
            continue
            
        c_polished, s_polished, val_polished = _polish_solution(c, s, deadline)
        
        if val_polished > best_val:
            best_val = val_polished
            best_centers = c_polished
            best_sides = s_polished
            
    # Final check and cleanup
    # The optimizer might leave the solution slightly invalid due to floating point
    # or if the penalty wasn't high enough.
    # We do a final high-precision polish on the best solution found.
    if time.time() < deadline - 10:
        c_final, s_final, val_final = _polish_solution(best_centers, best_sides, deadline)
        if val_final > best_val:
            best_val = val_final
            best_centers = c_final
            best_sides = s_final
            
    # If the best solution is still invalid, try to fix it
    if not _is_valid_vectorized(best_centers, best_sides, tol=1e-7):
        # Reduce sizes slightly to ensure validity
        # This is a fallback. Ideally, the high lambda optimization should have fixed it.
        # Let's try a few steps of gradient descent on the penalty only.
        c_fix = best_centers.copy()
        s_fix = best_sides.copy()
        
        # Reduce all sides by 1%
        s_fix *= 0.99
        # Re-center if needed? No, reducing size keeps center same, makes square smaller, 
        # so if it was inside, it's still inside. If it overlapped, it might not anymore.
        
        if _is_valid_vectorized(c_fix, s_fix, tol=1e-7):
            best_centers = c_fix
            best_sides = s_fix
            best_val = np.sum(s_fix)
        else:
            # Fallback to the known good grid
            best_centers, best_sides = _initial_packing_grid()
            best_val = 3.0

    return best_centers, best_sides

# EVOLVE-BLOCK-END
