import numpy as np

def moving_sofa_2d() -> np.ndarray:
    # Define the sofa points
    # i = 0..449, j = 0..299
    # x_sofa = -3 + 0.01*i
    # y_sofa = -1.5 + 0.01*j
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    sofa_points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Pre-compute base corridor check function
    # Base: {0 <= u <= 1, -3 <= v <= 1} U {-3 <= u <= 1, 0 <= v <= 1}
    def inside_base(points):
        u = points[:, 0]
        v = points[:, 1]
        mask1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        mask2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        return mask1 | mask2

    # Keyframe angles
    # We return 5 intermediate points.
    # Evaluator adds (0,0,0) and (-2,0,90).
    # Total 7 points.
    # Angles: 0, 15, 30, 45, 60, 75, 90
    angles = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]
    
    def evaluate_path_params(params):
        # params: x1, y1, x2, y2, x3, y3, x4, y4, x5, y5
        # Map to keyframes 1..5
        keyframes = np.array([
            [0.0, 0.0, 0.0],
            [params[0], params[1], 15.0],
            [params[2], params[3], 30.0],
            [params[4], params[5], 45.0],
            [params[6], params[7], 60.0],
            [params[8], params[9], 75.0],
            [-2.0, 0.0, 90.0]
        ])
        
        all_poses = []
        # Add first point
        all_poses.append(keyframes[0])
        
        for i in range(len(keyframes) - 1):
            p1 = keyframes[i]
            p2 = keyframes[i+1]
            da = p2[2] - p1[2]
            m = max(1, int(np.ceil(abs(da))))
            
            # Generate m interpolated points between p1 and p2
            # t_k = k / (m+1) for k=1..m
            if m >= 1:
                t_vals = np.linspace(0, 1, m+1)[1:-1]
                for t in t_vals:
                    x = p1[0] + t * (p2[0] - p1[0])
                    y = p1[1] + t * (p2[1] - p1[1])
                    a = p1[2] + t * (p2[2] - p1[2])
                    all_poses.append([x, y, a])
            
            # Add the next keyframe
            all_poses.append(p2)
            
        # Check intersection of sofa points
        valid_mask = np.ones(len(sofa_points), dtype=bool)
        
        for pose in all_poses:
            x, y, a_deg = pose
            a_rad = np.deg2rad(a_deg)
            
            # Rotate sofa points by -a_rad and translate by -x, -y
            # w' = R(-a) (w - c)
            # R(-a) = [[cos, sin], [-sin, cos]]
            
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            dx = sofa_points[:, 0] - x
            dy = sofa_points[:, 1] - y
            
            u = cos_a * dx + sin_a * dy
            v = -sin_a * dx + cos_a * dy
            
            mask = inside_base(np.column_stack([u, v]))
            valid_mask &= mask
            
            if not np.any(valid_mask):
                break # Early exit if no points remain
        
        return np.sum(valid_mask) * 0.0001

    # Simulated Annealing Optimization
    np.random.seed(42)
    
    # Initial guess: Linear interpolation in x, constant y=0.6?
    # Let's start with a reasonable heuristic path.
    # x(t) from 0 to -2, y(t) bulging up.
    # t in [0, 1] corresponding to angle 0..90
    # x = -2*t, y = 0.6 * sin(pi*t)
    # Angles: 15, 30, 45, 60, 75 -> t = 1/6, 1/3, 1/2, 2/3, 5/6
    t_vals = np.array([15, 30, 45, 60, 75]) / 90.0
    x_init = -2.0 * t_vals
    y_init = 0.6 * np.sin(np.pi * t_vals)
    
    current_params = np.concatenate([x_init, y_init])
    current_score = evaluate_path_params(current_params)
    best_params = current_params.copy()
    best_score = current_score
    
    # SA Parameters
    T = 1.0
    cooling_rate = 0.99
    min_T = 0.01
    max_iterations = 2000
    step_size = 0.2
    
    for i in range(max_iterations):
        # Perturb
        new_params = current_params + np.random.normal(0, step_size, size=len(current_params))
        
        # Clamp to reasonable bounds
        # x in [-3, 1], y in [-2, 2]
        new_params[::2] = np.clip(new_params[::2], -3.0, 1.0)
        new_params[1::2] = np.clip(new_params[1::2], -2.0, 2.0)
        
        new_score = evaluate_path_params(new_params)
        
        # Acceptance
        if new_score > current_score:
            current_params = new_params
            current_score = new_score
            if current_score > best_score:
                best_params = current_params.copy()
                best_score = current_score
        else:
            # Accept with probability exp((new - current) / T)
            # Score is area, higher is better. So delta = new - current < 0.
            # We want to accept worse solutions with decreasing probability.
            delta = new_score - current_score
            if delta < 0:
                prob = np.exp(delta / T)
                if np.random.random() < prob:
                    current_params = new_params
                    current_score = new_score
            
        # Cool
        T *= cooling_rate
        
        if T < min_T:
            break
            
    x1, y1, x2, y2, x3, y3, x4, y4, x5, y5 = best_params
    
    path = np.array([
        [x1, y1, 15.0],
        [x2, y2, 30.0],
        [x3, y3, 45.0],
        [x4, y4, 60.0],
        [x5, y5, 75.0]
    ])
    
    return path
