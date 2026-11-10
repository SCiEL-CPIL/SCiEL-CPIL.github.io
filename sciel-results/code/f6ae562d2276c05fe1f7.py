# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize as opt
from scipy.interpolate import CubicSpline

def moving_sofa_2d() -> np.ndarray:
    """
    Moves a sofa through an L-shaped corridor.
    Uses a cubic B-spline path parameterized by angle, optimized with Nelder-Mead.
    """
    # 1. Setup Grid
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

    # 2. Cubic B-spline path definition
    # We define x(t) and y(t) as cubic splines where t is in [0, 1]
    # corresponding to angle 0 to 90 degrees.
    # Control points: we use 5 knot points for each of x and y (including endpoints)
    # So coeffs = [x0, x1, x2, x3, x4, y0, y1, y2, y3, y4]
    # where x_i and y_i are the spline values at t_i = i/4 for i=0..4
    # The spline is constrained to pass through (0,0) at t=0 and (-2,0) at t=1
    
    N_KNOTS = 5  # number of control points per coordinate
    
    def get_path_from_coeffs(coeffs, n_waypoints=30):
        """
        coeffs: array of shape (10,) -> [x0..x4, y0..y4]
        Returns: np.ndarray of shape (n_waypoints, 3)
        """
        xs_ctrl = coeffs[:N_KNOTS]
        ys_ctrl = coeffs[N_KNOTS:]
        
        # Knot locations in [0, 1]
        t_knots = np.linspace(0, 1, N_KNOTS)
        
        # Create cubic splines
        try:
            cs_x = CubicSpline(t_knots, xs_ctrl)
            cs_y = CubicSpline(t_knots, ys_ctrl)
        except Exception:
            # Fallback to linear interpolation if spline fails
            t_sample = np.linspace(0, 1, n_waypoints)
            xs = np.interp(t_sample, t_knots, xs_ctrl)
            ys = np.interp(t_sample, t_knots, ys_ctrl)
            angles_deg = t_sample * 90.0
            # Exclude endpoints to avoid duplicates with start/end
            mask = (angles_deg > 0.5) & (angles_deg < 89.5)
            return np.column_stack((xs[mask], ys[mask], angles_deg[mask]))
        
        # Sample the spline at n_waypoints points
        t_sample = np.linspace(0, 1, n_waypoints)
        xs = cs_x(t_sample)
        ys = cs_y(t_sample)
        angles_deg = t_sample * 90.0
        
        # Exclude points too close to 0 or 90 to avoid duplicate/zero-step issues
        mask = (angles_deg > 0.5) & (angles_deg < 89.5)
        
        if not np.any(mask):
            return np.array([[-1.0, 0.1, 45.0]])
            
        return np.column_stack((xs[mask], ys[mask], angles_deg[mask]))

    def objective(coeffs):
        """
        Objective function for minimization (scipy minimizes, so we negate area).
        """
        path = get_path_from_coeffs(coeffs, n_waypoints=25)
        area = evaluate_path_fast(path)
        return -area

    # 3. Optimization
    start_time = time.time()
    deadline = start_time + 590.0
    
    # Initial guess: x goes from 0 to -2, y stays small
    # Control points for x: [0, -0.5, -1.0, -1.5, -2.0]
    # Control points for y: [0, 0.1, 0.2, 0.1, 0.0]
    init_coeffs = np.array([0.0, -0.5, -1.0, -1.5, -2.0, 0.0, 0.1, 0.2, 0.1, 0.0])
    
    best_coeffs = init_coeffs.copy()
    best_score = -objective(init_coeffs)
    
    # Try a few random perturbations to find a good starting point
    n_starts = 8
    for i in range(n_starts):
        if time.time() > deadline - 100:
            break
            
        if i == 0:
            coeffs = init_coeffs.copy()
        else:
            # Random perturbation around the initial guess
            scale = 0.3 / (i + 1)
            coeffs = init_coeffs + np.random.uniform(-scale, scale, size=10)
            # Ensure endpoints are reasonable
            coeffs[0] = np.clip(coeffs[0], -0.5, 0.5)
            coeffs[-1] = np.clip(coeffs[-1], -0.5, 0.5)
            coeffs[N_KNOTS] = np.clip(coeffs[N_KNOTS], -0.5, 0.5)
            coeffs[N_KNOTS + N_KNOTS - 1] = np.clip(coeffs[N_KNOTS + N_KNOTS - 1], -0.5, 0.5)
        
        score = -objective(coeffs)
        if score > best_score:
            best_score = score
            best_coeffs = coeffs.copy()
            
    # Run Nelder-Mead optimization
    if time.time() < deadline - 60:
        try:
            res = opt.minimize(objective, best_coeffs, method='Nelder-Mead', 
                               options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6,
                                        'adaptive': True})
            
            if time.time() < deadline - 30:
                final_score = -res.fun
                if final_score > best_score:
                    best_score = final_score
                    best_coeffs = res.x
        except Exception as e:
            pass
    
    # Final polish: tighter optimization with more waypoints
    if time.time() < deadline - 20:
        def objective_polish(coeffs):
            path = get_path_from_coeffs(coeffs, n_waypoints=40)
            area = evaluate_path_fast(path)
            return -area
        
        try:
            res2 = opt.minimize(objective_polish, best_coeffs, method='Nelder-Mead', 
                                options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-7,
                                         'adaptive': True})
            
            if time.time() < deadline - 5:
                final_score2 = -res2.fun
                if final_score2 > best_score:
                    best_score = final_score2
                    best_coeffs = res2.x
        except Exception as e:
            pass
    
    # 4. Final Path Construction
    if best_coeffs is not None:
        final_path = get_path_from_coeffs(best_coeffs, n_waypoints=35)
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
    if not np.all(np.isfinite(final_path)) or len(final_path) == 0:
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
