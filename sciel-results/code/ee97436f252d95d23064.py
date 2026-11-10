import numpy as np
import time
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline
import sys

def moving_sofa_2d() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 600s limit
    
    # Sofa grid points
    x_sofa = np.arange(-3, 1.5, 0.01)
    y_sofa = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(x_sofa, y_sofa)
    sofa_points = np.vstack([X.ravel(), Y.ravel()]).T
    n_pts = len(sofa_points)
    
    # Precompute for fast evaluation
    # We'll use a subsample for optimization, full set for final check
    
    def evaluate_path_fast(path, subsample=True):
        """Evaluate the area of the sofa for a given path.
        path: (N, 3) array of (x, y, a_degrees)
        Returns estimated area.
        """
        # Add endpoints
        if len(path) == 0:
            full_path = np.array([[0.0, 0.0, 0.0], [-2.0, 0.0, 90.0]])
        else:
            full_path = np.vstack([np.array([[0.0, 0.0, 0.0]]), path, np.array([[-2.0, 0.0, 90.0]])])
        
        # Densify the path according to evaluator rules
        # m = max(1, ceil(|a2 - a1|))
        dense_path = [full_path[0]]
        for i in range(len(full_path) - 1):
            p1 = full_path[i]
            p2 = full_path[i + 1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            if m > 1:
                for j in range(1, m):
                    t = j / m
                    interp = p1 + t * (p2 - p1)
                    dense_path.append(interp)
            dense_path.append(p2)
        
        dense_path = np.array(dense_path)
        
        # Use subsample for speed during optimization
        if subsample:
            idx = np.random.choice(n_pts, size=min(2000, n_pts), replace=False)
            pts = sofa_points[idx]
            scale = n_pts / len(idx)
        else:
            pts = sofa_points
            scale = 1.0
        
        # Check if each point is inside all placements
        # A point w is inside placement (x, y, a) if R(-a)(w - (x, y)) is in base corridor
        # Base corridor: {0<=u<=1, -3<=v<=1} ∪ {-3<=u<=1, 0<=v<=1}
        
        inside_all = np.ones(len(pts), dtype=bool)
        
        for row in dense_path:
            x_c, y_c, a_deg = row
            a_rad = np.radians(a_deg)
            
            dx = pts[:, 0] - x_c
            dy = pts[:, 1] - y_c
            
            # Rotate by -a
            cos_a = np.cos(a_rad)
            sin_a = np.sin(a_rad)
            
            # R(-a) = [[cos(a), sin(a)], [-sin(a), cos(a)]]
            u = cos_a * dx + sin_a * dy
            v = -sin_a * dx + cos_a * dy
            
            # Check base corridor
            mask1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            mask2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            mask = mask1 | mask2
            
            inside_all &= mask
            
            # Early termination if no points remain
            if not inside_all.any():
                break
        
        count = np.sum(inside_all)
        area = count * scale * 0.0001
        return area
    
    def evaluate_path_full(path):
        """Full evaluation without subsampling."""
        return evaluate_path_fast(path, subsample=False)
    
    # Parametric path representation using cubic B-spline
    # We'll define the path as a function of angle θ from 0 to 90 degrees
    # x(θ) and y(θ) are cubic B-splines with control points
    
    def spline_path_to_points(ctrl_x, ctrl_y, n_samples=100):
        """Convert spline control points to a path array.
        ctrl_x, ctrl_y: control points for x and y as functions of θ (in degrees)
        Returns (N, 3) array.
        """
        # θ values from 0 to 90
        theta_vals = np.linspace(0, 90, n_samples)
        
        # Create cubic splines
        # Control points should be at evenly spaced θ values
        n_ctrl = len(ctrl_x)
        ctrl_theta = np.linspace(0, 90, n_ctrl)
        
        # Ensure monotonic and valid for spline
        try:
            cs_x = CubicSpline(ctrl_theta, ctrl_x, bc_type='natural')
            cs_y = CubicSpline(ctrl_theta, ctrl_y, bc_type='natural')
        except Exception:
            # Fallback to linear interpolation
            cs_x = np.interp(theta_vals, ctrl_theta, ctrl_x)
            cs_y = np.interp(theta_vals, ctrl_theta, ctrl_y)
            return np.column_stack([cs_x, cs_y, theta_vals])
        
        x_vals = cs_x(theta_vals)
        y_vals = cs_y(theta_vals)
        
        return np.column_stack([x_vals, y_vals, theta_vals])
    
    def objective(params):
        """Objective function for optimization.
        params: flattened array of control points [x_ctrl, y_ctrl]
        Returns negative area (to minimize).
        """
        n_ctrl = len(params) // 2
        ctrl_x = params[:n_ctrl]
        ctrl_y = params[n_ctrl:]
        
        # Time check
        if time.time() > deadline - 10:
            return 0.0
        
        # Convert to path
        path = spline_path_to_points(ctrl_x, ctrl_y, n_samples=50)
        
        # Evaluate
        area = evaluate_path_fast(path, subsample=True)
        
        return -area
    
    # Initial guess: a smooth arc
    # Control points for x and y as functions of θ (0 to 90 degrees)
    n_ctrl = 6
    ctrl_theta = np.linspace(0, 90, n_ctrl)
    
    # Initial guess: x goes from 0 to -2, y has a bump
    # x(θ) ≈ -2 * (θ/90)^2 (quadratic)
    # y(θ) ≈ 1.5 * sin(π * θ/90) (sine bump)
    ctrl_x_init = -2.0 * (ctrl_theta / 90.0)**2
    ctrl_y_init = 1.5 * np.sin(np.pi * ctrl_theta / 90.0)
    
    # Ensure endpoints are reasonable
    ctrl_x_init[0] = 0.0  # Start at (0,0)
    ctrl_x_init[-1] = -2.0  # End at (-2,0)
    ctrl_y_init[0] = 0.0
    ctrl_y_init[-1] = 0.0
    
    params_init = np.concatenate([ctrl_x_init, ctrl_y_init])
    
    # Run SLSQP optimization
    best_params = params_init.copy()
    best_area = evaluate_path_fast(spline_path_to_points(ctrl_x_init, ctrl_y_init, n_samples=50), subsample=True)
    
    # Try multiple random restarts
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 60:
            break
        
        if restart == 0:
            params = params_init.copy()
        else:
            # Perturb initial guess
            params = params_init + 0.3 * np.random.randn(len(params_init))
            params[0] = 0.0  # Keep start fixed
            params[-1] = -2.0  # Keep end fixed
            params[n_ctrl] = 0.0
            params[-1 + n_ctrl] = 0.0
        
        try:
            result = minimize(
                objective,
                params,
                method='SLSQP',
                options={'maxiter': 200, 'ftol': 1e-6, 'disp': False}
            )
            
            if result.success or (result.fun < -best_area):
                if -result.fun > best_area:
                    best_area = -result.fun
                    best_params = result.x.copy()
        except Exception as e:
            pass
        
        # Check time
        if time.time() > deadline - 30:
            break
    
    # Extract best path
    ctrl_x_best = best_params[:n_ctrl]
    ctrl_y_best = best_params[n_ctrl:]
    
    # Generate final path with more samples
    final_path = spline_path_to_points(ctrl_x_best, ctrl_y_best, n_samples=200)
    
    # Remove the first point (0,0,0) and last point (-2,0,90) since evaluator adds them
    # Actually, let's keep them to be safe, but the evaluator will add its own
    # The evaluator adds (0,0,0) before and (-2,0,90) after
    # So we should not include exact duplicates, but it should be fine
    
    # Trim to avoid exact endpoint duplicates
    # First point of final_path is (0, 0, 0) approximately
    # Last point is (-2, 0, 90) approximately
    # Let's remove the first and last to avoid redundancy
    if len(final_path) > 2:
        final_path = final_path[1:-1]
    
    # Ensure all values are finite
    if not np.all(np.isfinite(final_path)):
        # Fallback to a simple linear path
        final_path = np.array([
            [-0.5, 0.5, 22.5],
            [-1.0, 1.0, 45.0],
            [-1.5, 0.5, 67.5]
        ])
    
    return final_path
