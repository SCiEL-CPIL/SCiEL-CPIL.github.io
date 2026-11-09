import numpy as np
import time
import math

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = 610.0  # Safety margin before 630s kill
    
    # Grid for evaluation, matching the evaluator
    # i = 0..449 -> x = -3 + 0.01i
    # j = 0..299 -> y = -1.5 + 0.01j
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    P = np.column_stack((X.ravel(), Y.ravel()))
    n_points = P.shape[0]
    
    def get_base_mask(u, v):
        # Base corridor:
        # Arm 1: 0 <= u <= 1, -3 <= v <= 1
        # Arm 2: -3 <= u <= 1, 0 <= v <= 1
        return ((u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)) | \
               ((u >= -3) & (u <= 1) & (v >= 0) & (v <= 1))

    def evaluate_path(path):
        """
        Calculate the area of the intersection of the corridor placements.
        path: array of shape (n, 3)
        """
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
        
        valid = np.ones(n_points, dtype=bool)
        
        for px, py, pa in dense_path:
            a_rad = np.deg2rad(pa)
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            dx = P[:, 0] - px
            dy = P[:, 1] - py
            
            u = cos_a * dx + sin_a * dy
            v = -sin_a * dx + cos_a * dy
            
            in_base = get_base_mask(u, v)
            
            valid = valid & in_base
            
            if not np.any(valid):
                return 0.0
                
        return np.sum(valid) * 0.0001

    # Strategy:
    # The path is defined by points (x(t), y(t), a(t)).
    # The evaluator interpolates linearly between our provided points.
    # If we provide points at every integer degree from 1 to 89, the interpolation
    # between them is a single step (m=1 for 1 degree diff? No, m=ceil(1)=1, so just the endpoint).
    # Wait, if we provide a=1, a=2, ..., the evaluator sees delta a = 1.
    # m = max(1, ceil(1)) = 1.
    # So it just adds the next point.
    # This means if we provide 89 points (1 to 89), the dense path is exactly those 89 points
    # plus the endpoints (0,0,0) and (-2,0,90).
    # Total 91 placements.
    
    # This allows us to control the position (x,y) at every integer angle.
    # We want to maximize the area of the intersection of the "safe zones" for all these angles.
    
    # A point (X, Y) is valid if for all k in 0..90, (X,Y) is in the corridor at placement k.
    # Let's pre-compute the set of valid points for each angle if we could choose the optimal (x,y) for that angle.
    # But (x,y) is coupled across angles.
    
    # Alternative:
    # The intersection of corridors is a convex set? No, corridors are convex, intersection is convex.
    # Actually, the set of points that can be occupied by the sofa is the intersection of the "reachable" areas.
    
    # Let's use a heuristic:
    # The optimal path for the "moving sofa" problem in a corridor is known to be a specific curve.
    # For a unit width corridor, the path of the inner corner is roughly an arc.
    # Let's try to optimize the 89 points (x1, y1, ..., x89, y89) using a simple local search / gradient-free method
    # but with a much better initialization and a more efficient evaluation.
    
    # To speed up evaluation, we can early-exit if the valid set becomes empty.
    # Also, we can vectorize the check for a single angle.
    
    # Let's try a different approach:
    # Instead of optimizing 178 variables, let's assume the path is smooth and parameterize it by a few control points
    # but use a higher fidelity evaluator.
    
    # Actually, the previous program used 5 points. Let's try 20 points (every 5 degrees: 5, 10, ..., 85).
    # This reduces the variable count to 40.
    # The evaluator will interpolate between them.
    # If we use 5-degree steps, m = ceil(5) = 5.
    # So between angle 5 and 10, it adds 4 intermediate points.
    # This is manageable.
    
    # Let's use 18 points: 5, 10, ..., 90? No, 90 is fixed at end.
    # Let's use points at 5, 10, ..., 85. (17 points)
    # Plus endpoints 0 and 90.
    # Total segments: 18.
    
    angles = np.arange(5, 90, 5) # 5, 10, ..., 85. 17 points.
    n_ctrl = len(angles)
    
    def objective(params):
        if time.time() > deadline - 5:
            return 0.0
        
        # params: x1, y1, x2, y2, ...
        path = np.zeros((n_ctrl, 3))
        for i in range(n_ctrl):
            path[i, 0] = params[2*i]
            path[i, 1] = params[2*i+1]
            path[i, 2] = angles[i]
            
        return -evaluate_path(path)

    # Initial guess:
    # The sofa moves from (0,0) to (-2,0).
    # A common approximation for the path of the corner is:
    # x(a) = -2 * a / 90
    # y(a) = h * sin(pi * a / 90) ?
    # Let's refine this.
    
    x0 = np.zeros(2 * n_ctrl)
    for i, a in enumerate(angles):
        t = a / 90.0
        x0[2*i] = -2 * t
        # Height of the sofa path?
        # The maximum height is constrained by the corridor width.
        # Let's guess a sine wave with amplitude 1.
        x0[2*i+1] = math.sin(math.pi * t)
        
    # Use Nelder-Mead
    from scipy.optimize import minimize
    
    best_result = None
    best_score = -np.inf
    
    try:
        result = minimize(
            objective, 
            x0, 
            method='Nelder-Mead', 
            options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-5, 'disp': False}
        )
        
        if time.time() < deadline - 10:
            score = -result.fun
            if score > best_score:
                best_score = score
                best_result = result.x
                
    except Exception as e:
        pass
        
    # If we have time, try a second phase with more points or finer search?
    # The 17-point control is likely sufficient for a smooth path.
    
    if best_result is None:
        # Fallback
        best_result = x0
        
    # Construct final path
    final_path = np.zeros((n_ctrl, 3))
    for i in range(n_ctrl):
        final_path[i, 0] = best_result[2*i]
        final_path[i, 1] = best_result[2*i+1]
        final_path[i, 2] = angles[i]
        
    return final_path
