import numpy as np
import time
import math
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = 615.0  # Safety margin before 630s kill
    
    # Grid for evaluation, matching the evaluator
    # i = 0..449 -> x = -3 + 0.01i
    # j = 0..299 -> y = -1.5 + 0.01j
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    P = np.column_stack((X.ravel(), Y.ravel()))
    n_points = P.shape[0]
    
    # Pre-compute angles for the dense path evaluation
    # The evaluator interpolates between consecutive rows.
    # If we provide a path with angles 0, 1, 2, ..., 90, the interpolation is trivial (m=1 for each step).
    # But the evaluator prepends (0,0,0) and appends (-2,0,90).
    # If our path starts at angle 0 and ends at 90, the full path is 0..90.
    # Let's define the path in terms of a parameter t in [0, 1] corresponding to angle a = 90*t.
    
    def get_dense_path_from_params(params):
        """
        params: [x1, y1, x2, y2]
        Control points for a quadratic Bezier curve:
        P0 = (0, 0)
        P2 = (-2, 0)
        P1 = (x1, y1)
        
        Wait, a quadratic Bezier has 3 control points. P0 and P2 are fixed.
        So we only have P1 = (x1, y1). That's 2 parameters.
        Is that enough? A quadratic Bezier is a parabola.
        The optimal path is often approximated by a circular arc or a specific curve.
        A quadratic Bezier might be too restrictive.
        Let's use a Cubic Bezier: P0=(0,0), P3=(-2,0), P1=(x1,y1), P2=(x2,y2).
        This gives 4 parameters.
        
        B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3
        """
        x1, y1, x2, y2 = params
        
        # We need to generate the path points that we return.
        # The task says: "returning a float array of shape (n, 3), n >= 1"
        # "Row (x, y, a) places the width-1 L-shaped corridor..."
        # "The evaluator puts (0, 0, 0) before your rows and (-2, 0, 90) after them."
        # "Between consecutive rows it adds linear interpolations of (x, y, a) in m = max(1, ceil(|a2 - a1|)) equal steps"
        
        # To maximize efficiency and accuracy, we should return points at integer angles 1 to 89.
        # This ensures m=1 for all segments, so no interpolation is needed by the evaluator.
        # The evaluator will just use our points as is, with the endpoints prepended/appended.
        
        angles = np.arange(1, 90)
        t_vals = angles / 90.0
        
        # Compute Bezier curve points
        # P0 = (0,0), P3 = (-2,0)
        # P1 = (x1, y1), P2 = (x2, y2)
        
        # B(t) x-coord:
        # 3(1-t)^2 t x1 + 3(1-t) t^2 x2 + t^3 (-2)
        # B(t) y-coord:
        # 3(1-t)^2 t y1 + 3(1-t) t^2 y2
        
        # Vectorized computation
        t = t_vals
        mt = 1 - t
        
        bx = 3 * mt**2 * t * x1 + 3 * mt * t**2 * x2 + t**3 * (-2.0)
        by = 3 * mt**2 * t * y1 + 3 * mt * t**2 * y2
        
        path = np.column_stack([bx, by, angles])
        return path

    def evaluate_path_fast(path, active_mask=None):
        """
        Calculate the area of the intersection of the corridor placements.
        path: array of shape (n, 3)
        active_mask: optional boolean mask of points to check (from previous step)
        Returns: (score, new_active_mask)
        """
        if not isinstance(path, np.ndarray):
            path = np.array(path)
            
        # Prepend and append fixed endpoints
        full_path = np.vstack([np.array([0.0, 0.0, 0.0]), path, np.array([-2.0, 0.0, 90.0])])
        
        # Densify path exactly as evaluator does
        dense_path = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i+1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            if m == 1:
                dense_path.append(p2)
            else:
                for k in range(1, m+1):
                    t = k / m
                    interp = p1 + t * (p2 - p1)
                    dense_path.append(interp)
        
        if active_mask is None:
            valid = np.ones(n_points, dtype=bool)
        else:
            valid = active_mask.copy()
            
        if not np.any(valid):
            return 0.0, valid
            
        # Loop through dense_path and update valid
        for px, py, pa in dense_path:
            a_rad = np.deg2rad(pa)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            dx = P[:, 0] - px
            dy = P[:, 1] - py
            
            u = cos_a * dx + sin_a * dy
            v = -sin_a * dx + cos_a * dy
            
            in_base = ((u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)) | \
                      ((u >= -3) & (u <= 1) & (v >= 0) & (v <= 1))
            
            valid = valid & in_base
            
            # Early exit if too few points remain
            count = np.sum(valid)
            if count < 100:
                return 0.0, valid
                
        score = np.sum(valid) * 0.0001
        return score, valid

    # Objective function for SciPy
    # We want to maximize score, so we minimize -score.
    def objective(params):
        path = get_dense_path_from_params(params)
        score, _ = evaluate_path_fast(path)
        return -score

    # Initial guess
    # A good initial guess is a circular arc or a simple parabola.
    # Let's try a symmetric parabola: y = h * (1 - (x+1)^2) ? No, x goes from 0 to -2.
    # Midpoint x=-1. y(0)=0, y(-2)=0.
    # y = h * (1 - (x+1)^2)
    # For a Bezier, P1 and P2 should be chosen to approximate this.
    # Let's start with P1 = (-0.5, 1.0), P2 = (-1.5, 1.0). This creates a hump.
    init_params = np.array([-0.5, 1.0, -1.5, 1.0])
    
    # Bounds
    # x1, x2 should be between -2 and 0.
    # y1, y2 should be between 0 and 2.
    bounds = [(-2.0, 0.0), (0.0, 2.0), (-2.0, 0.0), (0.0, 2.0)]
    
    # Run optimization
    best_result = None
    best_score = -np.inf
    
    # We can run multiple optimizations with different initial guesses or just one L-BFGS-B.
    # L-BFGS-B is robust. Let's try it.
    
    try:
        res = minimize(objective, init_params, method='L-BFGS-B', bounds=bounds, 
                       options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.success or res.fun < best_score:
            best_result = res
            best_score = res.fun
            
        # If time permits, try a few other initial guesses
        if time.time() < deadline - 60:
            other_inits = [
                [-1.0, 1.2, -1.0, 1.2], # Symmetric
                [-0.8, 1.5, -1.2, 1.5], # Asymmetric
                [-0.5, 0.8, -1.5, 0.8], # Lower
                [-1.0, 1.8, -1.0, 1.8], # Higher
            ]
            for init in other_inits:
                if time.time() > deadline - 30:
                    break
                res = minimize(objective, np.array(init), method='L-BFGS-B', bounds=bounds, 
                               options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
                if res.fun < best_score:
                    best_score = res.fun
                    best_result = res
    except Exception:
        pass
    
    # If optimization failed or didn't find anything, use the initial guess
    if best_result is None:
        final_params = init_params
    else:
        final_params = best_result.x
        
    # Generate final path
    final_path = get_dense_path_from_params(final_params)
    
    # Final evaluation to ensure it's valid and get the score
    final_score, _ = evaluate_path_fast(final_path)
    
    # Ensure output format is correct: float array of shape (n, 3)
    # The task says: returning a float array of shape (n, 3), n >= 1, all finite.
    # Row (x, y, a)
    
    return final_path
