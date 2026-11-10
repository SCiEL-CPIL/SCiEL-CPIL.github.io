# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize

def moving_sofa_2d() -> np.ndarray:
    """
    Moves a sofa through an L-shaped corridor.
    Strategy: Optimize a set of free control points (waypoints) that define the path.
    The path is a sequence of points (x_i, y_i, a_i). 
    We optimize the x_i and y_i for a fixed set of angles a_i.
    This allows for a much more flexible path shape than a low-degree polynomial.
    """
    
    # 1. Setup Grid
    x_grid = np.arange(-3, 1.5, 0.01)
    y_grid = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(x_grid, y_grid)
    P = np.column_stack((X.ravel(), Y.ravel()))
    N_points = P.shape[0]
    wx = P[:, 0]
    wy = P[:, 1]
    
    # We define a fixed set of angles at which we place control points.
    # The evaluator interpolates between our returned rows.
    # To maximize control, we should return rows at specific angles.
    # The evaluator adds (0,0,0) and (-2,0,90).
    # Let's choose a set of intermediate angles.
    # If we return points at angles A = [a_1, ..., a_k], the evaluator will interpolate
    # between (0,0,0) and (x_1, y_1, a_1), then (x_1, y_1, a_1) and (x_2, y_2, a_2), etc.
    # To ensure the sofa is inside the corridor during interpolation, we need the path to be smooth.
    # Let's use a dense set of angles for the control points to effectively define the path.
    # However, optimizing too many points is expensive.
    # Let's try optimizing ~20-30 waypoints.
    
    # Angles for our control points (excluding 0 and 90)
    # We can choose these adaptively or just use a uniform grid for the angles,
    # and only optimize x and y.
    n_waypoints = 20
    a_ctrl = np.linspace(1, 89, n_waypoints)
    
    # Variables to optimize: x_i, y_i for i in 1..n_waypoints
    # Total dimensions: 2 * n_waypoints
    
    def get_path_from_params(params):
        """
        params: array of shape (2 * n_waypoints,)
        Returns path array of shape (n_waypoints, 3)
        """
        x_ctrl = params[:n_waypoints]
        y_ctrl = params[n_waypoints:]
        path = np.column_stack((x_ctrl, y_ctrl, a_ctrl))
        return path

    def evaluate_area(params):
        """
        Computes the area of the sofa for the path defined by params.
        We must simulate the evaluator's interpolation to be accurate.
        Evaluator:
        1. Starts at (0,0,0)
        2. For each row in our output:
           - Interpolate from prev to current
           - Check all interpolated steps
        3. Ends at (-2,0,90)
        """
        path = get_path_from_params(params)
        
        # Build the full sequence of states including start and end
        # Start
        states = [np.array([0.0, 0.0, 0.0])]
        # Intermediate
        for i in range(n_waypoints):
            states.append(path[i])
        # End
        states.append(np.array([-2.0, 0.0, 90.0]))
        
        mask = np.ones(N_points, dtype=bool)
        
        # Process segments
        for i in range(len(states) - 1):
            s1 = states[i]
            s2 = states[i+1]
            
            a1 = s1[2]
            a2 = s2[2]
            
            # Number of interpolation steps
            # m = max(1, ceil(|a2 - a1|))
            da = abs(a2 - a1)
            if da < 1e-9:
                m = 1
            else:
                m = max(1, int(np.ceil(da)))
                
            # If m > 1, we have intermediate steps
            # If m == 1, we just have the endpoint s2 (s1 is already checked as endpoint of prev seg or start)
            # Actually, the evaluator adds linear interpolations in m equal steps.
            # This usually means m-1 intermediate points plus the endpoint?
            # "adds linear interpolations of (x, y, a) in m = ... equal steps"
            # Standard interpretation: step size = 1/m of the delta.
            # Points: s1 + k/m * (s2 - s1) for k=1..m.
            # So we check m new points.
            
            for k in range(1, m + 1):
                t = k / m
                x = s1[0] + t * (s2[0] - s1[0])
                y = s1[1] + t * (s2[1] - s1[1])
                a_deg = s1[2] + t * (s2[2] - s1[2])
                
                a_rad = np.deg2rad(a_deg)
                cos_a = np.cos(a_rad)
                sin_a = np.sin(a_rad)
                
                dx = wx - x
                dy = wy - y
                
                u = dx * cos_a + dy * sin_a
                v = -dx * sin_a + dy * cos_a
                
                in_r1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
                in_r2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
                
                in_corridor = in_r1 | in_r2
                
                mask &= in_corridor
                
                # Early exit
                if not np.any(mask):
                    return 0.0
        
        return float(np.sum(mask)) * 0.0001

    # Initial guess
    # A simple linear path in x and y is a bad start for the "turning" part.
    # Let's start with a path that approximates the known optimal shape roughly.
    # x goes from 0 to -2.
    # y stays near 0 but might need to move.
    # Let's try a simple curve.
    
    # Since a is uniform in our control points, we can just initialize x,y linearly.
    x_init = np.linspace(0, -2, n_waypoints)
    y_init = np.zeros(n_waypoints)
    
    # Add a small perturbation to help escape local minima if symmetric
    np.random.seed(42)
    y_init += np.random.uniform(-0.1, 0.1, n_waypoints)
    
    initial_params = np.concatenate([x_init, y_init])
    
    best_score = 0.0
    best_params = initial_params
    
    start_time = time.time()
    deadline = start_time + 580.0
    
    # Optimization
    # Nelder-Mead is good for low dimensions (40 dims here might be slow but feasible)
    # Let's try a few restarts.
    
    n_restarts = 5
    
    for r in range(n_restarts):
        if time.time() > deadline:
            break
            
        if r == 0:
            params = initial_params
        else:
            # Perturb the initial guess
            params = initial_params + np.random.normal(0, 0.5, size=initial_params.shape)
            
        try:
            res = scipy.optimize.minimize(
                lambda p: -evaluate_area(p),
                params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-6, 'adaptive': True}
            )
            score = -res.fun
            if score > best_score:
                best_score = score
                best_params = res.x
        except Exception as e:
            pass
            
    # Generate final path from best_params
    final_path = get_path_from_params(best_params)
    
    # Ensure finite
    if not np.all(np.isfinite(final_path)):
        final_path = np.array([[0.0, 0.0, 45.0]])
        
    return final_path

# EVOLVE-BLOCK-END
