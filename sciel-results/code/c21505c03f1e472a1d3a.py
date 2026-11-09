# EVOLVE-BLOCK-START
import numpy as np
import time


def _base_inside(u, v):
    """Check if point (u,v) is in the base L-shaped corridor."""
    # Base: {0<=u<=1, -3<=v<=1} UNION {-3<=u<=1, 0<=v<=1}
    in_arm1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
    in_arm2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
    return in_arm1 | in_arm2


def _eval_path(path):
    """
    Compute the sofa area for a given path.
    path: (n, 3) array of (x, y, angle_degrees)
    The evaluator prepends (0,0,0) and appends (-2,0,90), then densifies.
    Returns: area of the sofa (number of grid points * 0.0001)
    """
    # Build the full path with start and end
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([-2.0, 0.0, 90.0])
    
    full_path = np.vstack([start, path, end])
    
    # Densify: between consecutive rows, add linear interpolations
    # m = max(1, ceil(|a2 - a1|)) equal steps
    densified = [full_path[0]]
    for i in range(len(full_path) - 1):
        p1 = full_path[i]
        p2 = full_path[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da)))
        if m == 1:
            densified.append(p2)
        else:
            for k in range(1, m + 1):
                t = k / m
                interp = p1 + t * (p2 - p1)
                densified.append(interp)
    
    densified = np.array(densified)
    
    # Sofa grid: points (-3 + 0.01i, -1.5 + 0.01j), i=0..449, j=0..299
    i_vals = np.arange(0, 450)
    j_vals = np.arange(0, 300)
    xs = -3.0 + 0.01 * i_vals
    ys = -1.5 + 0.01 * j_vals
    
    # Create grid of all sofa candidate points
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points = np.column_stack([X.ravel(), Y.ravel()])
    n_points = points.shape[0]
    
    # For each placement, check which points are inside
    # A point w is inside placement (x,y,a) if R(-a)(w - (x,y)) is in base
    # R(-a) is rotation by -a degrees
    
    # Start with all points as candidates
    inside_all = np.ones(n_points, dtype=bool)
    
    for placement in densified:
        x, y, a_deg = placement
        a_rad = np.radians(a_deg)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        # R(-a) * (w - (x,y))
        # R(-a) = [[cos(a), sin(a)], [-sin(a), cos(a)]]
        dx = points[:, 0] - x
        dy = points[:, 1] - y
        
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        
        in_base = _base_inside(u, v)
        inside_all &= in_base
        
        # Early termination if no points remain
        if not inside_all.any():
            break
    
    count = np.sum(inside_all)
    return count * 0.0001


def moving_sofa_2d() -> np.ndarray:
    """
    Find a path that maximizes the sofa area.
    Returns: np.ndarray of shape (n, 3)
    """
    deadline = time.time() + 600  # 600 second limit
    
    # Start with the baseline path
    baseline = np.array(
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
    
    best_path = baseline
    best_area = _eval_path(baseline)
    
    # Try to improve using local search / random perturbations
    rng = np.random.default_rng(42)
    
    # We'll try various strategies:
    # 1. Small random perturbations of the baseline
    # 2. Try different numbers of points
    # 3. Try structured paths
    
    max_iter = 200
    for it in range(max_iter):
        if time.time() > deadline - 30:
            break
        
        # Generate a candidate path by perturbing the baseline
        n_points = len(baseline)
        # Randomly choose number of points
        if rng.random() < 0.3:
            n_points = int(rng.integers(5, 12))
        
        if n_points == len(baseline):
            # Perturb the baseline
            candidate = baseline + rng.normal(0, 0.05, size=baseline.shape)
            # Ensure angles are in a reasonable range
            candidate[:, 2] = np.clip(candidate[:, 2], 5, 85)
            # Ensure x is in reasonable range
            candidate[:, 0] = np.clip(candidate[:, 0], -1.0, 0.5)
            candidate[:, 1] = np.clip(candidate[:, 1], -0.5, 0.8)
        else:
            # Generate a new path
            angles = np.linspace(10, 85, n_points) + rng.normal(0, 2, n_points)
            angles = np.clip(angles, 5, 85)
            xs = np.linspace(-0.1, -0.7, n_points) + rng.normal(0, 0.03, n_points)
            ys = np.linspace(0.2, 0.3, n_points) + rng.normal(0, 0.03, n_points)
            candidate = np.column_stack([xs, ys, angles])
        
        area = _eval_path(candidate)
        
        if area > best_area:
            best_area = area
            best_path = candidate
    
    return best_path


# EVOLVE-BLOCK-END
