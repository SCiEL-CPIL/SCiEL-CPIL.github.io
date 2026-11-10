# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline


def _base_contains(u, v):
    """Check if points (u, v) are inside the base L-shaped corridor.
    Base = union of {0<=u<=1, -3<=v<=1} and {-3<=u<=1, 0<=v<=1}
    """
    in_rect1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
    in_rect2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
    return in_rect1 | in_rect2


def _sofa_area(path):
    """Compute the sofa area for a given path (n, 3) array of (x, y, angle_deg).
    The evaluator prepends (0,0,0) and appends (-2,0,90), then densifies.
    We replicate that here for scoring.
    """
    # Prepend start and append end
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([-2.0, 0.0, 90.0])
    full_path = np.vstack([start, path, end])

    # Densify: between consecutive rows, add linear interpolations in m = max(1, ceil(|a2-a1|)) steps
    densified = [full_path[0]]
    for i in range(len(full_path) - 1):
        p1 = full_path[i]
        p2 = full_path[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da)))
        if m > 1:
            for k in range(1, m):
                t = k / m
                pt = p1 + t * (p2 - p1)
                densified.append(pt)
        densified.append(p2)
    densified = np.array(densified)

    # Sofa grid points
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    gx, gy = np.meshgrid(xs, ys, indexing='xy')
    grid_points = np.column_stack([gx.ravel(), gy.ravel()])

    # For each placement, check if all grid points are inside
    # A point w is inside placement (x,y,a) if R(-a)(w - (x,y)) is in base
    count = len(grid_points)
    inside_all = np.ones(count, dtype=bool)

    for (x, y, a_deg) in densified:
        a_rad = np.radians(a_deg)
        # R(-a) = rotation by -a
        c = np.cos(-a_rad)
        s = np.sin(-a_rad)
        # w - (x,y)
        dx = grid_points[:, 0] - x
        dy = grid_points[:, 1] - y
        # Rotate
        u = c * dx - s * dy
        v = s * dx + c * dy
        in_base = _base_contains(u, v)
        inside_all &= in_base
        if not inside_all.any():
            break

    area = inside_all.sum() * 0.0001
    return area


def moving_sofa_2d() -> np.ndarray:
    """
    Optimize a smooth B-spline path to maximize the moving sofa area.
    
    The path is parameterized by a cubic B-spline over a normalized parameter t in [0, 1].
    We optimize the control points of the spline (position and angle) using SLSQP.
    
    Returns:
        path: np.ndarray of shape (n, 3) with n >= 1, all finite.
    """
    # Define the spline with a larger number of control points for flexibility
    n_ctrl = 8
    t_knots = np.linspace(0, 1, n_ctrl)
    
    # Improved initial guess: 
    # The sofa moves from (0,0) to (-2,0) while rotating 0 to 90 deg.
    # A good heuristic is to move along an arc that keeps the sofa centered in the corridor.
    # We will create a smooth arc for x, y and linear for angle.
    
    # Approximate path points for initialization
    # We want the path to be roughly:
    # Start (0,0) -> Move left/up slightly -> Turn -> Move left -> End (-2,0)
    
    # Let's try a circular arc approximation for the center of the sofa
    # Radius ~ 1.5?
    # Let's just use a smooth interpolation of known good waypoints
    # Waypoints:
    # t=0: (0,0, 0)
    # t=0.2: (-0.2, 0.3, 20)
    # t=0.4: (-0.6, 0.4, 45)
    # t=0.6: (-1.0, 0.3, 60)
    # t=0.8: (-1.5, 0.1, 80)
    # t=1.0: (-2.0, 0.0, 90)
    
    # We only return the middle part, so we sample t in (0, 1)
    # But the control points define the spline over [0,1].
    # The evaluator adds (0,0,0) at start and (-2,0,90) at end.
    # So our spline should ideally start near (0,0,0) and end near (-2,0,90).
    # However, the spline is defined by control points.
    # If we set control points at t=0 and t=1 to be the start/end, the spline will pass through them (if using appropriate conditions).
    # CubicSpline with not-a-knot boundary conditions does not necessarily pass through the first/last control points exactly unless specified.
    # Actually, CubicSpline(t_knots, x) passes through (t_knots[i], x[i]).
    # So if we set x[0]=0, y[0]=0, a[0]=0 and x[-1]=-2, y[-1]=0, a[-1]=90, the spline will pass through start and end.
    # But we don't want to return the start and end, the evaluator adds them.
    # If we return points that include the start/end, the evaluator will duplicate them?
    # "The evaluator puts (0, 0, 0) before your rows and (-2, 0, 90) after them."
    # If my first row is (0,0,0), the full path starts with (0,0,0), (0,0,0)... which is fine, just redundant.
    # But to be safe, we should probably return points strictly between start and end.
    
    # Let's define control points that include the start and end, but when sampling, we skip the very first and last if they are too close to the boundaries added by the evaluator.
    # Or better: define the spline to go from Start to End, and sample interior points.
    
    x_ctrl = np.array([0.0, -0.2, -0.5, -0.8, -1.1, -1.4, -1.7, -2.0])
    y_ctrl = np.array([0.0, 0.2, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0])
    a_ctrl = np.array([0.0, 15.0, 35.0, 55.0, 70.0, 80.0, 88.0, 90.0])
    
    # Combine into a single parameter vector
    def unpack(params):
        x = params[0:n_ctrl]
        y = params[n_ctrl:2*n_ctrl]
        a = params[2*n_ctrl:3*n_ctrl]
        return x, y, a
    
    def make_path(params, n_samples=10):
        """Create a path by sampling the B-spline at n_samples points."""
        x, y, a = unpack(params)
        # Create cubic splines for each component
        # Use not-a-knot boundary conditions
        try:
            sx = CubicSpline(t_knots, x)
            sy = CubicSpline(t_knots, y)
            sa = CubicSpline(t_knots, a)
        except Exception:
            # Fallback to linear if spline fails
            t_samples = np.linspace(0.05, 0.95, n_samples)
            x_s = np.interp(t_samples, t_knots, x)
            y_s = np.interp(t_samples, t_knots, y)
            a_s = np.interp(t_samples, t_knots, a)
            return np.column_stack([x_s, y_s, a_s])
        
        # Sample the spline at n_samples points (excluding endpoints since evaluator adds them)
        # We want the intermediate points, so sample in (0, 1)
        # Avoid t=0 and t=1 to prevent duplication with evaluator's added points
        t_samples = np.linspace(0.05, 0.95, n_samples)
        x_s = sx(t_samples)
        y_s = sy(t_samples)
        a_s = sa(t_samples)
        
        # Ensure angles are within reasonable bounds
        a_s = np.clip(a_s, 0.1, 89.9)
        
        return np.column_stack([x_s, y_s, a_s])
    
    def objective(params):
        """Negative area to minimize."""
        path = make_path(params)
        # Ensure all values are finite
        if not np.all(np.isfinite(path)):
            return 1e10
        try:
            area = _sofa_area(path)
            return -area
        except Exception:
            return 1e10
    
    # Initial parameter vector
    x0 = np.concatenate([x_ctrl, y_ctrl, a_ctrl])
    
    # Bounds for control points
    # x: roughly -2.5 to 0.5
    # y: roughly -1.0 to 1.5
    # a: 0 to 90
    bounds = [(-2.5, 0.5)] * n_ctrl + [(-1.0, 1.5)] * n_ctrl + [(0.0, 90.0)] * n_ctrl
    
    # Constraints: ensure the path is roughly monotonic in angle (smooth turning)
    # And that positions are within reasonable bounds
    constraints = []
    
    # Ensure angle is monotonically increasing (smooth turn)
    for i in range(n_ctrl - 1):
        def mono_angle(params, i=i):
            _, _, a = unpack(params)
            return a[i+1] - a[i] - 0.0  # should be >= 0
        constraints.append({'type': 'ineq', 'fun': mono_angle})
    
    # Run SLSQP optimization
    # We'll run a few iterations with different initial guesses to avoid local minima
    
    best_params = x0
    best_area = -objective(x0)
    
    # Run optimization from initial guess
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 300, 'ftol': 1e-8, 'disp': False}
    )
    
    if result.success and -result.fun > best_area:
        best_params = result.x
        best_area = -result.fun
        
    # Try a few perturbations
    for trial in range(3):
        np.random.seed(42 + trial)
        # Small perturbations
        x_pert = x_ctrl + np.random.normal(0, 0.05, n_ctrl)
        y_pert = y_ctrl + np.random.normal(0, 0.05, n_ctrl)
        a_pert = a_ctrl + np.random.normal(0, 2.0, n_ctrl)
        
        # Ensure monotonic angle
        for i in range(1, n_ctrl):
            if a_pert[i] <= a_pert[i-1]:
                a_pert[i] = a_pert[i-1] + 0.5
        
        # Clip to bounds
        x_pert = np.clip(x_pert, -2.5, 0.5)
        y_pert = np.clip(y_pert, -1.0, 1.5)
        a_pert = np.clip(a_pert, 0.0, 90.0)
        
        params_pert = np.concatenate([x_pert, y_pert, a_pert])
        
        result_pert = minimize(
            objective,
            params_pert,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200, 'ftol': 1e-8, 'disp': False}
        )
        
        if result_pert.success and -result_pert.fun > best_area:
            best_area = -result_pert.fun
            best_params = result_pert.x
            
    # Create the final path
    path = make_path(best_params, n_samples=10)
    
    # Ensure all values are finite
    if not np.all(np.isfinite(path)):
        # Fallback to a simple path
        path = np.array([
            [-0.1, 0.1, 10.0],
            [-0.2, 0.2, 20.0],
            [-0.4, 0.3, 30.0],
            [-0.6, 0.3, 45.0],
            [-0.8, 0.2, 60.0],
            [-1.0, 0.1, 70.0],
            [-1.3, 0.05, 80.0],
            [-1.6, 0.0, 85.0],
        ], dtype=np.float64)
    
    # Clip angle to reasonable range
    path[:, 2] = np.clip(path[:, 2], 0.1, 89.9)
    
    return path.astype(np.float64)


# EVOLVE-BLOCK-END
