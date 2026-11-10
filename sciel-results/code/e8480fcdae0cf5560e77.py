# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _build_grid():
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    return X.ravel(), Y.ravel()

def _densify_path(path):
    full = np.vstack([[0.0, 0.0, 0.0], path, [-2.0, 0.0, 90.0]])
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

def _point_in_base(u, v):
    # Base L-shape: {0<=u<=1, -3<=v<=1} U {-3<=u<=1, 0<=v<=1}
    arm1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
    arm2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
    return arm1 | arm2

def _evaluate_path(path, X, Y):
    """
    Evaluate the area of the intersection of all corridor placements.
    Returns the raw value (number of sofa points * 0.0001).
    """
    densified = _densify_path(path)
    n_points = len(X)
    
    # For each point, check if it's inside all placements
    # A point w is inside placement (x, y, a) if R(-a)(w - (x, y)) lies in the base
    # R(-a) is rotation by -a degrees
    
    inside_all = np.ones(n_points, dtype=bool)
    
    for i in range(len(densified)):
        x, y, a_deg = densified[i]
        a_rad = np.deg2rad(a_deg)
        
        # Translation
        dx = X - x
        dy = Y - y
        
        # Rotation by -a_rad
        cos_a = np.cos(-a_rad)
        sin_a = np.sin(-a_rad)
        
        u = cos_a * dx - sin_a * dy
        v = sin_a * dx + cos_a * dy
        
        in_base = _point_in_base(u, v)
        inside_all &= in_base
        
        # Early exit if no points remain
        if not np.any(inside_all):
            break
    
    count = np.sum(inside_all)
    return count * 0.0001

def _negative_area(params, X, Y):
    """Objective function for minimization (we maximize area)."""
    path = params.reshape(-1, 3)
    area = _evaluate_path(path, X, Y)
    return -area

def moving_sofa_2d() -> np.ndarray:
    """
    The 2D moving-sofa problem: find the largest-area rigid shape that can be moved
    through an L-shaped corridor of width 1 that makes a single 90-degree turn.

    The shape is specified IMPLICITLY by a motion "path". Each path point is a triple
    (x, y, angle_degrees) giving the position of the corridor's inner corner and the
    corridor's rotation at one instant of the move. In the reference frame of the sofa
    the corridor sweeps over the plane; the sofa is then the intersection of every
    corridor placement along the path, so by construction it always clears the corridor.
    The objective is to maximize the area of that intersection.

    The evaluator prepends the start state (0, 0, 0) and appends the end state
    (-2, 0, 90), then densifies the path to <=1-degree rotation steps, exactly as in the
    AlphaEvolve "single turn" corridor notebook.

    Returns:
        path: np.ndarray of shape (n, 3); each row is (x, y, angle_degrees) for the
            corridor's inner corner. Here n = 8. Endpoints are added by the evaluator.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # 600 second limit
    
    # Build the grid of sofa points
    X, Y = _build_grid()
    
    # Start with the baseline path
    best_path = np.array(
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
    best_area = _evaluate_path(best_path, X, Y)
    
    # Try optimization with different initializations
    n_points = len(X)
    
    # Try a few random initializations
    np.random.seed(42)
    
    for restart in range(20):
        # Check deadline
        if time.time() > deadline - 60.0:
            break
        
        # Generate a random initial path
        # Angles should be monotonically increasing from ~15 to ~86
        angles = np.sort(np.random.uniform(10, 90, 8))
        # Ensure monotonic increase
        for i in range(1, len(angles)):
            angles[i] = max(angles[i], angles[i-1] + 5)
        
        # Positions: x from -0.1 to -0.7, y from 0.1 to 0.3
        xs = np.linspace(-0.1, -0.7, 8) + np.random.uniform(-0.1, 0.1, 8)
        ys = np.linspace(0.1, 0.3, 8) + np.random.uniform(-0.1, 0.1, 8)
        
        init_path = np.column_stack([xs, ys, angles])
        
        # Flatten for scipy
        init_params = init_path.ravel()
        
        # Optimize using L-BFGS-B
        try:
            result = minimize(
                _negative_area,
                init_params,
                args=(X, Y),
                method='L-BFGS-B',
                options={'maxiter': 50, 'ftol': 1e-6, 'gtol': 1e-4}
            )
            
            if result.success or result.fun < -best_area:
                opt_path = result.x.reshape(-1, 3)
                opt_area = -result.fun
                
                if opt_area > best_area:
                    best_area = opt_area
                    best_path = opt_path.copy()
        except Exception:
            pass
    
    # Return the best path found
    return best_path

# EVOLVE-BLOCK-END
