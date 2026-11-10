# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

def _build_grid(subsample=1):
    """
    Precompute grid points for sofa membership testing.
    subsample: 1 for full grid, >1 for subsampled grid (for optimization speed).
    """
    # Full grid dimensions based on task description
    # i = 0..449, j = 0..299
    # x = -3 + 0.01i, y = -1.5 + 0.01j
    
    if subsample == 1:
        xs = np.arange(-3.0, 1.5, 0.01)
        ys = np.arange(-1.5, 1.5, 0.01)
    else:
        # Subsample to speed up optimization
        xs = np.arange(-3.0, 1.5, 0.01 * subsample)
        ys = np.arange(-1.5, 1.5, 0.01 * subsample)
        
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    return X.ravel(), Y.ravel(), len(xs) * len(ys)

def _densify_path(path):
    """
    Densify a path the same way the evaluator does.
    Prepends (0,0,0) and appends (-2,0,90).
    Between consecutive rows, adds linear interpolations in m = max(1, ceil(|a2-a1|)) steps.
    """
    # Ensure path is numpy array
    path = np.array(path, dtype=np.float64)
    if len(path) == 0:
        return np.array([[0.0, 0.0, 0.0], [-2.0, 0.0, 90.0]])
    
    # Prepend start, append end
    full = np.vstack([[0.0, 0.0, 0.0], path, [-2.0, 0.0, 90.0]])
    
    # Densify
    result = [full[0]]
    for i in range(len(full) - 1):
        p1 = full[i]
        p2 = full[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da)))
        for k in range(1, m + 1):
            t = k / m
            interp = p1 + t * (p2 - p1)
            result.append(interp)
    return np.array(result)

def _evaluate_path_fast(path, grid_x, grid_y, n_points):
    """
    Evaluate the sofa area for a given path.
    Uses vectorized operations.
    """
    densified = _densify_path(path)
    
    # Start with all points as potential sofa members
    inside_all = np.ones(n_points, dtype=bool)
    
    for (cx, cy, a_deg) in densified:
        a_rad = np.radians(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        # R(-a) * (w - c)
        dx = grid_x - cx
        dy = grid_y - cy
        
        # R(-a) = [[cos(a), sin(a)], [-sin(a), cos(a)]]
        # u = dx*cos(a) + dy*sin(a)
        # v = -dx*sin(a) + dy*cos(a)
        u = dx * cos_a + dy * sin_a
        v = -dx * sin_a + dy * cos_a
        
        # Check if in base L-shape
        # Arm 1: 0 <= u <= 1, -3 <= v <= 1
        # Arm 2: -3 <= u <= 1, 0 <= v <= 1
        in_base = ((u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)) | \
                  ((u >= -3) & (u <= 1) & (v >= 0) & (v <= 1))
        
        inside_all &= in_base
        
        # Early exit if no points remain
        if not np.any(inside_all):
            return 0.0
    
    return float(np.sum(inside_all))

def moving_sofa_2d() -> np.ndarray:
    """
    Optimize the moving sofa path to maximize the sofa area.
    """
    start_time = time.time()
    time_limit = 600  # Leave margin for safety (limit is 630)
    
    # Build full grid for final evaluation
    full_grid_x, full_grid_y, full_n = _build_grid(1)
    
    # Build subsampled grid for optimization
    # Subsample factor 3 reduces points by ~9x
    sub_grid_x, sub_grid_y, sub_n = _build_grid(3)
    
    def objective(params):
        """
        Objective function for optimization.
        params: flattened array of [x1, y1, dtheta1, x2, y2, dtheta2, ...]
        """
        # Time check
        if time.time() - start_time > time_limit - 10:
            return 0.0
        
        # Unpack parameters
        # We define N intermediate points.
        # Let's use 12 intermediate points to allow flexible shaping.
        N = 12
        # Each point has x, y, dtheta. dtheta is the angle increment from the previous point.
        # Total angle must be 90.
        # We enforce this by normalizing dthetas if they don't sum to 90? 
        # Or better: parameterize dthetas as positive values and scale them to sum to 90.
        
        if len(params) != N * 3:
            return 0.0
            
        pts = params.reshape(N, 3)
        xs = pts[:, 0]
        ys = pts[:, 1]
        dthetas = pts[:, 2]
        
        # Ensure dthetas are positive
        dthetas = np.maximum(dthetas, 0.1)
        
        # Scale dthetas to sum to 90
        sum_dtheta = np.sum(dthetas)
        if sum_dtheta == 0:
            dthetas = np.ones(N)
        else:
            dthetas = dthetas * (90.0 / sum_dtheta)
            
        # Construct the path
        angles = np.cumsum(dthetas)
        # The path points are (x, y, angle)
        # Note: The evaluator interpolates between these points.
        # We should provide the points that the evaluator will use as "rows".
        path = np.column_stack((xs, ys, angles))
        
        # Evaluate on subsampled grid
        score = _evaluate_path_fast(path, sub_grid_x, sub_grid_y, sub_n)
        
        # We want to maximize score, so minimize negative score
        return -score

    # Initial guess
    # A linear interpolation of position and angle is a safe start.
    # x: 0 to -2, y: 0 to 0, a: 0 to 90
    N = 12
    xs_init = np.linspace(0, -2, N+2)[1:-1]
    ys_init = np.zeros(N)
    # dthetas: equal steps
    dthetas_init = np.full(N, 90.0 / N)
    
    init_params = np.column_stack((xs_init, ys_init, dthetas_init)).ravel()
    
    # Optimization using Nelder-Mead
    # SciPy is available
    try:
        from scipy.optimize import minimize
        
        # Bounds: 
        # x: -3 to 1
        # y: -2 to 2
        # dtheta: 0.1 to 10 (will be scaled)
        bounds = [(-3.0, 1.0), (-2.0, 2.0), (0.1, 10.0)] * N
        
        result = minimize(
            objective, 
            init_params, 
            method='Nelder-Mead', 
            bounds=bounds,
            options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True}
        )
        
        best_params = result.x
        best_score_sub = -result.fun
        
    except Exception as e:
        print(f"Optimization failed: {e}", file=sys.stderr)
        best_params = init_params
        best_score_sub = objective(init_params)
        
    # Reconstruct best path from parameters
    pts = best_params.reshape(N, 3)
    xs = pts[:, 0]
    ys = pts[:, 1]
    dthetas = np.maximum(pts[:, 2], 0.1)
    sum_dtheta = np.sum(dthetas)
    dthetas = dthetas * (90.0 / sum_dtheta)
    angles = np.cumsum(dthetas)
    best_path = np.column_stack((xs, ys, angles))
    
    # Final evaluation on full grid to get true score
    # Note: The evaluator will densify this path.
    # We should return the path points.
    
    # One more local refinement on the full grid? 
    # It might be too slow. Let's just return the optimized path.
    # The score is determined by the evaluator.
    
    # Ensure the path is valid and finite
    if not np.all(np.isfinite(best_path)):
        best_path = init_params.reshape(N, 3) # Fallback
        
    return best_path

# EVOLVE-BLOCK-END
