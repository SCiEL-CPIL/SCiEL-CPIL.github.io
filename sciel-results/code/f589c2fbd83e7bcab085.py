# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _evaluate_path_fast(path):
    """
    Evaluate the number of sofa grid points that remain inside the corridor
    for all placements along the given path (including start and end states).
    Returns the count (integer).
    """
    # Sofa grid points
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    sofa_pts = np.stack([X.ravel(), Y.ravel()], axis=1)  # (N, 2)
    N = sofa_pts.shape[0]
    
    # Full path with start and end
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([-2.0, 0.0, 90.0])
    full = np.vstack([start, path, end])
    
    # Densify: between consecutive rows, add linear interpolations
    # m = max(1, ceil(|a2 - a1|)) equal steps
    densified = [full[0]]
    for i in range(len(full) - 1):
        a1 = full[i, 2]
        a2 = full[i+1, 2]
        m = max(1, int(np.ceil(abs(a2 - a1))))
        for k in range(1, m + 1):
            t = k / m
            x = full[i, 0] + t * (full[i+1, 0] - full[i, 0])
            y = full[i, 1] + t * (full[i+1, 1] - full[i, 1])
            a = a1 + t * (a2 - a1)
            densified.append(np.array([x, y, a]))
    
    densified = np.array(densified)  # (M, 3)
    M = densified.shape[0]
    
    # Base corridor: union of {0<=u<=1, -3<=v<=1} and {-3<=u<=1, 0<=v<=1}
    # A point w is inside placement (x,y,a) if R(-a)(w - (x,y)) is in base.
    # R(-a) rotates by -a degrees.
    
    # For each placement, check all sofa points.
    # We need ALL placements to contain the point.
    # Process placements in batches for memory efficiency.
    
    batch_size = 200
    all_inside = np.ones(N, dtype=bool)
    
    for b in range(0, M, batch_size):
        batch = densified[b:b+batch_size]  # (B, 3)
        B = batch.shape[0]
        
        for j in range(B):
            x, y, a_deg = batch[j]
            a_rad = np.radians(a_deg)
            ca = np.cos(-a_rad)
            sa = np.sin(-a_rad)
            
            # R(-a) * (w - (x,y))
            dx = sofa_pts[:, 0] - x
            dy = sofa_pts[:, 1] - y
            u = ca * dx - sa * dy
            v = sa * dx + ca * dy
            
            # Check if (u,v) is in base corridor
            # Base: {0<=u<=1, -3<=v<=1} OR {-3<=u<=1, 0<=v<=1}
            in_rect1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in_rect2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            in_base = in_rect1 | in_rect2
            
            all_inside &= in_base
            
            # Early exit if no points remain
            if not all_inside.any():
                return 0
    
    return int(all_inside.sum())


def _neg_count(params, n_ctrl):
    """Objective: negative count of sofa points (to minimize)."""
    # params: 3*n_ctrl values: x0,y0,a0, x1,y1,a1, ...
    path = params.reshape(n_ctrl, 3)
    # Ensure angles are in reasonable range
    path[:, 2] = np.clip(path[:, 2], 0.0, 90.0)
    # Ensure x,y in reasonable range
    path[:, 0] = np.clip(path[:, 0], -3.0, 0.0)
    path[:, 1] = np.clip(path[:, 1], -1.0, 1.0)
    
    count = _evaluate_path_fast(path)
    return -count


def moving_sofa_2d() -> np.ndarray:
    """
    Optimize the path for the 2D moving sofa problem using Nelder-Mead.
    Returns a float array of shape (n, 3) with n >= 1, all finite.
    """
    t_start = time.time()
    deadline = t_start + 580.0  # Safety margin before 630s limit
    
    # Number of control points
    n_ctrl = 20
    
    # Initial guess: interpolate from start to end
    angles = np.linspace(0.0, 90.0, n_ctrl + 2)[1:-1]  # exclude endpoints
    # Arc-like path for x,y
    t = np.linspace(0, 1, n_ctrl)
    x_init = -0.1 - 0.6 * t
    y_init = 0.1 + 0.2 * np.sin(np.pi * t)
    
    params0 = np.column_stack([x_init, y_init, angles]).ravel()
    
    best_path = params0.reshape(n_ctrl, 3)
    best_count = _evaluate_path_fast(best_path)
    
    # Nelder-Mead optimization
    try:
        result = minimize(
            _neg_count,
            params0,
            args=(n_ctrl,),
            method='Nelder-Mead',
            options={
                'maxiter': 500,
                'xatol': 0.01,
                'fatol': 1.0,
                'adaptive': True,
            }
        )
        if result.x is not None:
            candidate = result.x.reshape(n_ctrl, 3)
            candidate[:, 2] = np.clip(candidate[:, 2], 0.0, 90.0)
            candidate[:, 0] = np.clip(candidate[:, 0], -3.0, 0.0)
            candidate[:, 1] = np.clip(candidate[:, 1], -1.0, 1.0)
            cand_count = _evaluate_path_fast(candidate)
            if cand_count > best_count:
                best_count = cand_count
                best_path = candidate
    except Exception:
        pass
    
    # Check time, try a second optimization with different init if time permits
    elapsed = time.time() - t_start
    if elapsed < deadline - 120:
        # Second restart with perturbed initial
        np.random.seed(42)
        x_init2 = -0.1 - 0.6 * t + np.random.normal(0, 0.05, n_ctrl)
        y_init2 = 0.1 + 0.2 * np.sin(np.pi * t) + np.random.normal(0, 0.05, n_ctrl)
        a_init2 = angles + np.random.normal(0, 2.0, n_ctrl)
        params0_2 = np.column_stack([x_init2, y_init2, a_init2]).ravel()
        
        try:
            result2 = minimize(
                _neg_count,
                params0_2,
                args=(n_ctrl,),
                method='Nelder-Mead',
                options={
                    'maxiter': 400,
                    'xatol': 0.01,
                    'fatol': 1.0,
                    'adaptive': True,
                }
            )
            if result2.x is not None:
                candidate2 = result2.x.reshape(n_ctrl, 3)
                candidate2[:, 2] = np.clip(candidate2[:, 2], 0.0, 90.0)
                candidate2[:, 0] = np.clip(candidate2[:, 0], -3.0, 0.0)
                candidate2[:, 1] = np.clip(candidate2[:, 1], -1.0, 1.0)
                cand_count2 = _evaluate_path_fast(candidate2)
                if cand_count2 > best_count:
                    best_count = cand_count2
                    best_path = candidate2
        except Exception:
            pass
    
    # Final safety: ensure all values are finite
    best_path = np.nan_to_num(best_path, nan=0.0, posinf=0.0, neginf=0.0)
    
    return best_path


# EVOLVE-BLOCK-END
