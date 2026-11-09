import numpy as np
import time
import math
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = 620.0  # Safety margin before 630s kill
    
    # Grid for evaluation, matching the evaluator
    # i = 0..449 -> x = -3 + 0.01i
    # j = 0..299 -> y = -1.5 + 0.01j
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    # Flatten for vectorized operations
    P = np.column_stack((X.ravel(), Y.ravel()))
    n_points = P.shape[0]
    
    def evaluate_path(path):
        """
        Calculate the area of the intersection of the corridor placements.
        path: array of shape (n, 3)
        """
        # The evaluator puts (0,0,0) before and (-2,0,90) after.
        # We construct the full path including these endpoints.
        # Note: The task says "Row (x, y, a) places...".
        # The evaluator adds linear interpolations between consecutive rows.
        
        full_path = np.vstack([np.array([0.0, 0.0, 0.0]), path, np.array([-2.0, 0.0, 90.0])])
        
        # Densify path
        # The evaluator adds m = max(1, ceil(|a2 - a1|)) steps.
        # We simulate this exactly.
        
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
        
        # Check if all points in P are inside all placements
        valid = np.ones(n_points, dtype=bool)
        
        for px, py, pa in dense_path:
            a_rad = np.deg2rad(pa)
            # R(-a) rotation
            # u = cos(a)*dx + sin(a)*dy
            # v = -sin(a)*dx + cos(a)*dy
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            dx = P[:, 0] - px
            dy = P[:, 1] - py
            
            u = cos_a * dx + sin_a * dy
            v = -sin_a * dx + cos_a * dy
            
            # Base corridor:
            # Arm 1: 0 <= u <= 1, -3 <= v <= 1
            # Arm 2: -3 <= u <= 1, 0 <= v <= 1
            in_base = ((u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)) | \
                      ((u >= -3) & (u <= 1) & (v >= 0) & (v <= 1))
            
            valid = valid & in_base
            
            if not np.any(valid):
                return 0.0
                
        return np.sum(valid) * 0.0001

    def objective(params):
        """
        Params define the control points of the path.
        We use a set of 5 intermediate waypoints (x, y) and assume angle is
        linearly interpolated or defined by a smooth curve.
        
        To make it robust, we parameterize the path by 5 waypoints (x1,y1), ..., (x5,y5).
        The angles are determined by a smooth transition from 0 to 90.
        However, the evaluator interpolates (x,y,a) linearly between rows.
        So if we provide rows with specific angles, the evaluator will interpolate x,y,a linearly between them.
        
        Strategy:
        Provide a set of N rows.
        Let's fix N=5 rows for the optimizer.
        The angles will be roughly evenly spaced to ensure good coverage.
        Let angles be: 15, 30, 45, 60, 75.
        Or better, let the optimizer adjust the angles slightly too?
        No, fixed angles ensure the "densification" is predictable and covers the rotation.
        If angles are not monotonic, the path might sweep back and forth, which is usually bad.
        Let's fix angles to [15, 30, 45, 60, 75] to start, and maybe allow small perturbations?
        Actually, the position (x,y) at a given angle is the key.
        
        Let's parameterize:
        p = [x1, y1, x2, y2, x3, y3, x4, y4, x5, y5]
        Angles are fixed at [15, 30, 45, 60, 75].
        
        Wait, the evaluator adds (0,0,0) at start and (-2,0,90) at end.
        So the path segments are:
        (0,0,0) -> (x1,y1,15)
        (x1,y1,15) -> (x2,y2,30)
        ...
        (x5,y5,75) -> (-2,0,90)
        
        This is a well-posed optimization problem.
        """
        # time check
        if time.time() > deadline - 5:
            return 0.0
            
        x1, y1, x2, y2, x3, y3, x4, y4, x5, y5 = params
        
        # Define the path rows
        # We want the path to be smooth.
        # The "Sofa Curve" for a corridor of width 1 typically has the inner corner moving along a specific path.
        # Initial guess:
        # x: moves from 0 to -2 roughly.
        # y: moves up then down?
        
        path = np.array([
            [x1, y1, 15.0],
            [x2, y2, 30.0],
            [x3, y3, 45.0],
            [x4, y4, 60.0],
            [x5, y5, 75.0]
        ])
        
        score = evaluate_path(path)
        # Minimize negative score
        return -score

    # Initial guess based on typical sofa problem solutions
    # The inner corner (x,y) moves.
    # At a=0, corner is at (0,0).
    # At a=90, corner is at (-2,0).
    # In between, it arcs.
    # A common approximation:
    # x(t) = -2 * t
    # y(t) = 1 * sin(pi * t) ?
    
    # Let's try a guess:
    # t=0.16 (15 deg): x ~ -0.3, y ~ 0.3
    # t=0.33 (30 deg): x ~ -0.6, y ~ 0.6
    # t=0.50 (45 deg): x ~ -1.0, y ~ 0.8
    # t=0.66 (60 deg): x ~ -1.3, y ~ 0.6
    # t=0.83 (75 deg): x ~ -1.6, y ~ 0.3
    
    x0 = np.array([
        -0.3, 0.3,
        -0.6, 0.6,
        -1.0, 0.8,
        -1.3, 0.6,
        -1.6, 0.3
    ])
    
    # Use Nelder-Mead for deterministic optimization
    # This is robust for non-differentiable or noisy functions (though this is smooth-ish)
    
    best_result = None
    best_score = -np.inf
    
    # Run optimization
    try:
        result = minimize(
            objective, 
            x0, 
            method='Nelder-Mead', 
            options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-5, 'disp': False}
        )
        
        if time.time() < deadline - 10:
            score = -result.fun
            if score > best_score:
                best_score = score
                best_result = result.x
                
    except Exception as e:
        print(f"Optimization error: {e}")
        
    # If optimization failed or didn't improve, try a few random restarts with Nelder-Mead
    # Or just use the result.
    
    if best_result is None:
        # Fallback to seed-like path
        best_result = np.array([
            -0.1, 0.1,
            -0.2, 0.2,
            -0.3, 0.3,
            -0.4, 0.3,
            -0.5, 0.2
        ])
        
    # Construct final path from best_result
    x1, y1, x2, y2, x3, y3, x4, y4, x5, y5 = best_result
    
    final_path = np.array([
        [x1, y1, 15.0],
        [x2, y2, 30.0],
        [x3, y3, 45.0],
        [x4, y4, 60.0],
        [x5, y5, 75.0]
    ], dtype=np.float64)
    
    # One last check: if we have time, do a fine-tuning with more points?
    # The current path has 5 points. The evaluator interpolates.
    # We could add more points to allow more complex shapes.
    # But 5 points with linear interpolation in (x,y,a) might be restrictive if the optimal path is curved in x-y space for a given a.
    # However, the evaluator interpolates x,y linearly between rows.
    # So the path in 3D space is piecewise linear.
    # To get a smoother curve, we need more rows.
    
    # Let's try to optimize with 9 rows (angles 10, 20, ..., 80) if we have time.
    if time.time() < deadline - 60:
        def objective_9(params):
            if time.time() > deadline - 5:
                return 0.0
            # params: 9 * 2 = 18 values
            path = []
            angles = np.arange(10, 90, 10)
            for i in range(9):
                x = params[2*i]
                y = params[2*i+1]
                path.append([x, y, angles[i]])
            path = np.array(path)
            score = evaluate_path(path)
            return -score

        # Initial guess for 9 points
        # Interpolate the 5-point solution to 9 points
        # Angles: 10, 20, 30, 40, 50, 60, 70, 80
        # Our 5 points were at 15, 30, 45, 60, 75.
        
        # Simple linear interpolation of the 5-point path to 9 points
        # x(t), y(t) for t in [0,1] corresponding to angle 0..90
        # Our points:
        # a=15: (x1,y1)
        # a=30: (x2,y2)
        # a=45: (x3,y3)
        # a=60: (x4,y4)
        # a=75: (x5,y5)
        
        # We can just use the 5-point result as a starting point for the 9-point one by repeating/interpolating
        # Or just run Nelder-Mead again with a good random start near the previous solution.
        
        # Let's just use the 5-point solution. It's likely close to optimal for the "linear interpolation" constraint.
        # Adding more points allows for "kinks" which might not be optimal for a smooth corridor intersection.
        
    return final_path
