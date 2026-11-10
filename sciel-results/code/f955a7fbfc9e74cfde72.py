# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.interpolate import BSpline

# Time budget management
_START_TIME = time.time()
TIME_LIMIT = 580.0  # Safety margin before 600s evaluator kill


def _base_corridor_contains(u, v):
    """Check if point (u, v) is in the base L-shaped corridor (a=0).
    Base corridor = {0<=u<=1, -3<=v<=1} union {-3<=u<=1, 0<=v<=1}
    """
    in_arm1 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
    in_arm2 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
    return in_arm1 | in_arm2


def _sofa_grid():
    """Create the sofa grid points."""
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    XX, YY = np.meshgrid(xs, ys, indexing='xy')
    return XX.ravel(), YY.ravel()


def _evaluate_path(path, grid_x, grid_y, max_steps=200):
    """
    Evaluate the path: return the number of sofa grid points that lie inside
    every placement along the densified path.
    
    path: (n, 3) array of (x, y, angle_deg)
    """
    # Evaluator prepends (0,0,0) and appends (-2,0,90)
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([-2.0, 0.0, 90.0])
    
    # Build full sequence
    full = np.vstack([start, path, end])
    
    # Densify: between consecutive rows, add linear interpolations in
    # m = max(1, ceil(|a2 - a1|)) equal steps
    densified = [full[0]]
    for i in range(len(full) - 1):
        a1 = full[i, 2]
        a2 = full[i + 1, 2]
        m = max(1, int(np.ceil(abs(a2 - a1))))
        for step in range(1, m + 1):
            t = step / m
            pt = full[i] * (1 - t) + full[i + 1] * t
            densified.append(pt)
    
    densified = np.array(densified)
    
    # Cap the number of placements for speed
    if len(densified) > max_steps:
        # Uniformly subsample
        idx = np.linspace(0, len(densified) - 1, max_steps, dtype=int)
        densified = densified[idx]
    
    # For each placement, compute which sofa points are inside
    # A point w is inside placement (x, y, a) if R(-a)(w - (x,y)) is in base
    # R(-a) is rotation by -a degrees
    
    n_points = len(grid_x)
    inside_all = np.ones(n_points, dtype=bool)
    
    for i in range(len(densified)):
        x, y, a_deg = densified[i]
        a_rad = np.radians(a_deg)
        cos_a = np.cos(-a_rad)
        sin_a = np.sin(-a_rad)
        
        # Translate and rotate: u = cos(-a)*(wx - x) - sin(-a)*(wy - y)
        #                      v = sin(-a)*(wx - x) + cos(-a)*(wy - y)
        dx = grid_x - x
        dy = grid_y - y
        u = cos_a * dx - sin_a * dy
        v = sin_a * dx + cos_a * dy
        
        in_base = _base_corridor_contains(u, v)
        inside_all &= in_base
        
        # Early exit if no points remain
        if not inside_all.any():
            break
    
    return int(inside_all.sum())


def _score_from_count(count):
    """Convert point count to score (area * 10000)."""
    return count * 0.0001


def _bspline_path(control_points, t_eval):
    """
    Generate path points from B-spline control points.
    control_points: (n_ctrl, 3) array
    t_eval: array of parameter values
    Returns (n_eval, 3) array of (x, y, angle)
    """
    n_ctrl = len(control_points)
    # B-spline of order 4 (cubic)
    k = 3
    # Knot vector: open uniform
    knots = np.concatenate([np.zeros(k + 1), np.linspace(0, 1, n_ctrl - 2 * k + 1), np.ones(k + 1)])
    
    # Ensure knot vector is valid
    if len(knots) != n_ctrl + k + 1:
        # Fallback: use clamped uniform
        n_knots = n_ctrl + k + 1
        knots = np.zeros(n_knots)
        interior = np.linspace(0, 1, n_knots - 2 * k - 2)
        knots[k + 1:-k - 1] = interior
    
    # Evaluate for x, y, angle separately
    x_vals = []
    y_vals = []
    a_vals = []
    for t in t_eval:
        # Clamp t to [0, 1]
        tc = np.clip(t, 0, 1)
        # B-spline evaluation
        # Use scipy's BSpline
        for dim, cp in enumerate(control_points.T):
            try:
                bs = BSpline(knots, cp, k)
                val = bs(tc)
            except:
                # Fallback: simple interpolation
                if dim == 0:
                    x_vals.append(np.interp(tc, np.linspace(0, 1, n_ctrl), cp))
                elif dim == 1:
                    y_vals.append(np.interp(tc, np.linspace(0, 1, n_ctrl), cp))
                else:
                    a_vals.append(np.interp(tc, np.linspace(0, 1, n_ctrl), cp))
        if dim == 0:
            x_vals.append(val)
        elif dim == 1:
            y_vals.append(val)
        else:
            a_vals.append(val)
    
    return np.column_stack([x_vals, y_vals, a_vals])


def _objective(params, grid_x, grid_y, n_ctrl, t_eval, max_steps):
    """
    Objective function for optimization (negative score to maximize).
    params: flat array of control point coordinates
    """
    cp = params.reshape(n_ctrl, 3)
    path = _bspline_path(cp, t_eval)
    
    # Ensure path is finite
    if not np.all(np.isfinite(path)):
        return 1e10
    
    count = _evaluate_path(path, grid_x, grid_y, max_steps)
    return -count


def _optimize_bspline(grid_x, grid_y, n_ctrl, n_eval, max_steps, time_budget):
    """
    Optimize B-spline control points to maximize the sofa area.
    """
    t_eval = np.linspace(0.05, 0.95, n_eval)
    
    # Initial control points: interpolate between start (0,0,0) and end (-2,0,90)
    # with a smooth arc
    t_ctrl = np.linspace(0, 1, n_ctrl)
    cp_init = np.zeros((n_ctrl, 3))
    # Smooth path from (0,0,0) to (-2,0,90)
    # Use a gentle arc
    for i in range(n_ctrl):
        t = t_ctrl[i]
        # x: from 0 to -2
        cp_init[i, 0] = -2.0 * t
        # y: slight arc to help with the turn
        cp_init[i, 1] = 0.5 * np.sin(np.pi * t)
        # angle: from 0 to 90
        cp_init[i, 2] = 90.0 * t
    
    # Evaluate initial
    path_init = _bspline_path(cp_init, t_eval)
    count_init = _evaluate_path(path_init, grid_x, grid_y, max_steps)
    best_count = count_init
    best_params = cp_init.ravel().copy()
    
    # SLSQP optimization
    try:
        result = minimize(
            _objective,
            cp_init.ravel(),
            args=(grid_x, grid_y, n_ctrl, t_eval, max_steps),
            method='SLSQP',
            options={'maxiter': 50, 'ftol': 1e-6, 'disp': False}
        )
        if result.success and np.all(np.isfinite(result.x)):
            cp_opt = result.x.reshape(n_ctrl, 3)
            path_opt = _bspline_path(cp_opt, t_eval)
            count_opt = _evaluate_path(path_opt, grid_x, grid_y, max_steps)
            if count_opt > best_count:
                best_count = count_opt
                best_params = result.x.copy()
    except:
        pass
    
    # Try Nelder-Mead as well
    try:
        result2 = minimize(
            _objective,
            cp_init.ravel(),
            args=(grid_x, grid_y, n_ctrl, t_eval, max_steps),
            method='Nelder-Mead',
            options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-4, 'disp': False}
        )
        if np.all(np.isfinite(result2.x)):
            cp_opt2 = result2.x.reshape(n_ctrl, 3)
            path_opt2 = _bspline_path(cp_opt2, t_eval)
            count_opt2 = _evaluate_path(path_opt2, grid_x, grid_y, max_steps)
            if count_opt2 > best_count:
                best_count = count_opt2
                best_params = result2.x.copy()
    except:
        pass
    
    # Return best path
    best_cp = best_params.reshape(n_ctrl, 3)
    best_path = _bspline_path(best_cp, t_eval)
    return best_path, best_count


def _direct_optimize(grid_x, grid_y, n_points, max_steps, time_budget):
    """
    Directly optimize the path points using Nelder-Mead.
    """
    # Initial path: smooth arc
    t = np.linspace(0.05, 0.95, n_points)
    path_init = np.zeros((n_points, 3))
    for i in range(n_points):
        path_init[i, 0] = -2.0 * t[i]
        path_init[i, 1] = 0.5 * np.sin(np.pi * t[i])
        path_init[i, 2] = 90.0 * t[i]
    
    count_init = _evaluate_path(path_init, grid_x, grid_y, max_steps)
    best_count = count_init
    best_path = path_init.copy()
    
    def obj(params):
        path = params.reshape(n_points, 3)
        if not np.all(np.isfinite(path)):
            return 1e10
        return -_evaluate_path(path, grid_x, grid_y, max_steps)
    
    try:
        result = minimize(
            obj,
            path_init.ravel(),
            method='Nelder-Mead',
            options={'maxiter': 200, 'xatol': 1e-3, 'fatol': 1e-3, 'disp': False}
        )
        if np.all(np.isfinite(result.x)):
            path_opt = result.x.reshape(n_points, 3)
            count_opt = _evaluate_path(path_opt, grid_x, grid_y, max_steps)
            if count_opt > best_count:
                best_count = count_opt
                best_path = path_opt
    except:
        pass
    
    return best_path, best_count


def moving_sofa_2d() -> np.ndarray:
    """
    The 2D moving-sofa problem: find the largest-area rigid shape that can be moved
    through an L-shaped corridor of width 1 that makes a single 90-degree turn.
    """
    # Create sofa grid
    grid_x, grid_y = _sofa_grid()
    n_points_grid = len(grid_x)
    
    # Start with the baseline path as fallback
    baseline_path = np.array(
        [
            [-0.1, 0.1, 15.0],
            [-0.2, 0.2, 20.0],
            [-0.3, 0.3, 34.0],
            [-0.3, 0.3, 40.0],
            [-0.4, 0.3, 50.0],
            [-0.5, 0.2, 55.0],
            [-0.6, 0.2, 75.0],
            [-0.7, 0.3, 86.0],
        ],
        dtype=np.float64,
    )
    baseline_count = _evaluate_path(baseline_path, grid_x, grid_y, max_steps=150)
    
    best_path = baseline_path
    best_count = baseline_count
    
    # Try B-spline optimization
    elapsed = time.time() - _START_TIME
    if elapsed < TIME_LIMIT - 100:
        try:
            # Use fewer grid points for speed during optimization
            # Subsample the grid
            n_sub = min(5000, n_points_grid)
            idx = np.random.choice(n_points_grid, n_sub, replace=False)
            gx_sub = grid_x[idx]
            gy_sub = grid_y[idx]
            
            # B-spline with 6 control points, 10 evaluation points
            path_bs, count_bs = _optimize_bspline(
                gx_sub, gy_sub, n_ctrl=6, n_eval=10, max_steps=100,
                time_budget=TIME_LIMIT - elapsed - 50
            )
            
            # Verify with full grid
            count_bs_full = _evaluate_path(path_bs, grid_x, grid_y, max_steps=150)
            if count_bs_full > best_count:
                best_count = count_bs_full
                best_path = path_bs
        except:
            pass
    
    # Try direct optimization with fewer points
    elapsed = time.time() - _START_TIME
    if elapsed < TIME_LIMIT - 50:
        try:
            n_sub = min(3000, n_points_grid)
            idx = np.random.choice(n_points_grid, n_sub, replace=False)
            gx_sub = grid_x[idx]
            gy_sub = grid_y[idx]
            
            path_dir, count_dir = _direct_optimize(
                gx_sub, gy_sub, n_points=8, max_steps=80,
                time_budget=TIME_LIMIT - elapsed - 20
            )
            
            count_dir_full = _evaluate_path(path_dir, grid_x, grid_y, max_steps=150)
            if count_dir_full > best_count:
                best_count = count_dir_full
                best_path = path_dir
        except:
            pass
    
    # Final time check
    elapsed = time.time() - _START_TIME
    if elapsed > TIME_LIMIT - 10:
        # Return baseline if we're running out of time
        return baseline_path
    
    # Ensure the path is valid
    if not np.all(np.isfinite(best_path)):
        return baseline_path
    
    return best_path


# EVOLVE-BLOCK-END
