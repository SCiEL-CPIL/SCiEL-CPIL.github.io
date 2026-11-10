# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt

def moving_sofa_2d() -> np.ndarray:
    """
    Moves a sofa through an L-shaped corridor.
    Uses a deterministic parametric optimization strategy.
    The path is defined by x(theta) and y(theta) as smooth functions of angle theta.
    We optimize the coefficients of these functions to maximize the area of the sofa.
    """
    # 1. Setup Grid
    # Sofa points: x in [-3, 1.5), y in [-1.5, 1.5), step 0.01
    x_grid = np.arange(-3, 1.5, 0.01)
    y_grid = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(x_grid, y_grid)
    P = np.column_stack((X.ravel(), Y.ravel()))
    N_points = P.shape[0]
    
    # Start and End states
    start_state = [0.0, 0.0, 0.0]
    end_state = [-2.0, 0.0, 90.0]
    
    def densify_path(path_list):
        """
        Densifies a list of [x, y, a] such that angle steps are <= 1 degree.
        """
        if not path_list:
            return [start_state, end_state]
            
        full_path = [start_state] + list(path_list) + [end_state]
        
        densified = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i + 1]
            
            a1 = p1[2]
            a2 = p2[2]
            
            da = a2 - a1
            steps = max(1, int(np.ceil(abs(da))))
            
            if steps == 1:
                densified.append(p2)
            else:
                dx = (p2[0] - p1[0]) / steps
                dy = (p2[1] - p1[1]) / steps
                da_step = da / steps
                
                for j in range(1, steps):
                    nx = p1[0] + j * dx
                    ny = p1[1] + j * dy
                    na = a1 + j * da_step
                    densified.append([nx, ny, na])
                densified.append(p2)
                
        return densified

    def evaluate_path_fast(path_array):
        """
        Evaluates the area of the sofa for a given path array of shape (n, 3).
        Returns: float area
        """
        if path_array.size == 0:
            path_list = []
        else:
            path_list = path_array.tolist()
            
        dense_path = densify_path(path_list)
        
        mask = np.ones(N_points, dtype=bool)
        
        wx = P[:, 0]
        wy = P[:, 1]
        
        for p in dense_path:
            x, y, a_deg = p
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
            
            if not np.any(mask):
                return 0.0
                
        return float(np.sum(mask)) * 0.0001

    # 2. Parametric Path Definition
    # We define the path in terms of angle theta from 0 to 90 degrees.
    # We use a Fourier-like series for x and y to allow smooth, flexible curves.
    # x(theta) = c0 + c1*cos(t) + c2*sin(t) + c3*cos(2t) + c4*sin(2t)
    # y(theta) = d0 + d1*cos(t) + d2*sin(t) + d3*cos(2t) + d4*sin(2t)
    # where t = theta in radians.
    
    # Number of waypoints to sample from the parametric curve for evaluation
    N_WAYPOINTS = 20
    
    def get_path_from_coeffs(coeffs):
        """
        coeffs: array of shape (10,) -> [c0..c4, d0..d4]
        Returns: np.ndarray of shape (N_WAYPOINTS, 3)
        """
        c = coeffs[:5]
        d = coeffs[5:]
        
        # Sample angles uniformly from 0 to 90
        # We exclude 0 and 90 because the evaluator adds the start and end states.
        # However, the evaluator inserts the returned rows between start and end.
        # If we return rows with angles 0 or 90, it might cause duplicate or zero-step intervals.
        # It's safer to return angles strictly between 0 and 90.
        angles_deg = np.linspace(1, 89, N_WAYPOINTS)
        t = np.deg2rad(angles_deg)
        
        # Compute x, y
        cos_t = np.cos(t)
        sin_t = np.sin(t)
        cos_2t = np.cos(2 * t)
        sin_2t = np.sin(2 * t)
        
        xs = c[0] + c[1]*cos_t + c[2]*sin_t + c[3]*cos_2t + c[4]*sin_2t
        ys = d[0] + d[1]*cos_t + d[2]*sin_t + d[3]*cos_2t + d[4]*sin_2t
        
        # Construct path array
        path = np.column_stack((xs, ys, angles_deg))
        return path

    def objective(coeffs):
        """
        Objective function for minimization (scipy minimizes, so we negate area).
        """
        path = get_path_from_coeffs(coeffs)
        area = evaluate_path_fast(path)
        return -area

    # 3. Optimization
    start_time = time.time()
    deadline = start_time + 590.0
    
    # Initial guess based on known good paths (approximate)
    # The inner corner moves roughly from (0,0) to (-2,0) while rotating.
    # A simple linear interpolation in x,y with fixed angles is a bad start.
    # Let's use the seed path from the previous program as a guide for initial coefficients.
    # Seed path:
    # [-0.1, 0.1, 15.0],
    # [-0.2, 0.2, 20.0],
    # [-0.3, 0.3, 34.0],
    # [-0.3, 0.3, 40.0],
    # [-0.4, 0.3, 50.0],
    # [-0.5, 0.2, 55.0],
    # [-0.6, 0.2, 75.0],
    # [-0.7, 0.3, 86.0]
    
    # Fit these points to our parametric form to get a good initial guess?
    # Or just use a heuristic initial guess.
    # Let's try a few random initializations and pick the best one after a quick eval.
    
    best_coeffs = None
    best_score = -1.0
    
    # Try a few random starts
    n_starts = 5
    for i in range(n_starts):
        if time.time() > deadline:
            break
            
        if i == 0:
            # Heuristic start
            # x goes from ~0 to ~-2. y stays small.
            # c0 (bias) ~ -1
            # c1 (cos) ~ -1 (cos goes 1 to 0)
            # c2 (sin) ~ 0
            # c3 (cos2t) ~ 0
            # c4 (sin2t) ~ 0
            # y: d0 ~ 0.1, d1 ~ 0, d2 ~ 0, d3 ~ 0, d4 ~ 0
            coeffs = np.array([-1.0, -1.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.0])
        else:
            # Random start in reasonable bounds
            coeffs = np.random.uniform([-2, -2, -1, -1, -1, 0.5, -1, -1, -1, -1], 
                                       [0, 0, 1, 1, 1, 0.5, 1, 1, 1, 1])
            
        # Quick evaluation
        score = -objective(coeffs)
        if score > best_score:
            best_score = score
            best_coeffs = coeffs.copy()
            
    # If we found a good start, run a more intensive optimization
    if best_coeffs is not None:
        # Use Powell or Nelder-Mead
        # Powell is often better for non-differentiable or noisy functions, but here the function is discrete (grid points).
        # However, the parametric mapping is smooth. The evaluation is piecewise constant w.r.t coeffs.
        # So we are optimizing a step function. Gradient-based methods won't work well.
        # Derivative-free methods like Nelder-Mead or Powell are appropriate.
        
        # We can also increase the number of waypoints for final evaluation to ensure smoothness,
        # but for optimization, 20 waypoints might be enough to capture the shape.
        
        try:
            res = opt.minimize(objective, best_coeffs, method='Powell', 
                               options={'maxiter': 100, 'xatol': 1e-3, 'fatol': 1e-5})
            
            if time.time() < deadline:
                final_score = -res.fun
                if final_score > best_score:
                    best_score = final_score
                    best_coeffs = res.x
        except Exception as e:
            pass
            
    # 4. Final Path Construction
    if best_coeffs is not None:
        final_path = get_path_from_coeffs(best_coeffs)
    else:
        # Fallback to a simple valid path
        final_path = np.array([
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 20.0],
            [-0.3, 0.3, 34.0],
            [-0.3, 0.3, 40.0],
            [-0.4, 0.3, 50.0],
            [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 75.0],
            [-0.7, 0.3, 86.0],
        ], dtype=np.float64)
        
    # Ensure the path is valid (finite, correct shape)
    if not np.all(np.isfinite(final_path)):
        final_path = np.array([
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 20.0],
            [-0.3, 0.3, 34.0],
            [-0.3, 0.3, 40.0],
            [-0.4, 0.3, 50.0],
            [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 75.0],
            [-0.7, 0.3, 86.0],
        ], dtype=np.float64)
        
    return final_path

# EVOLVE-BLOCK-END
