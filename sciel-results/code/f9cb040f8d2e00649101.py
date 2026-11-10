# EVOLVE-BLOCK-START
import numpy as np
import time


def _is_inside_base(u, v):
    """Check if point (u, v) is in the base L-shaped corridor."""
    # Base corridor: union of {0<=u<=1, -3<=v<=1} and {-3<=u<=1, 0<=v<=1}
    return ((u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)) | \
           ((u >= -3) & (u <= 1) & (v >= 0) & (v <= 1))


def _compute_sofa_area(path, deadline=None):
    """
    Compute the area of the sofa (intersection of all corridor placements)
    for a given path. The evaluator adds (0,0,0) before and (-2,0,90) after.
    """
    # Build full placement list with evaluator's additions
    placements = np.vstack([
        np.array([[0.0, 0.0, 0.0]]),
        path,
        np.array([[-2.0, 0.0, 90.0]])
    ])
    
    # Densify: between consecutive rows, add linear interpolations
    # in m = max(1, ceil(|a2 - a1|)) equal steps
    densified = [placements[0]]
    for i in range(len(placements) - 1):
        p1 = placements[i]
        p2 = placements[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da)))
        if m > 1:
            for j in range(1, m):
                t = j / m
                interp = p1 + t * (p2 - p1)
                densified.append(interp)
        densified.append(p2)
    
    densified = np.array(densified)
    
    # Sofa grid points
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    sofa_points = np.column_stack([X.ravel(), Y.ravel()])
    n_points = len(sofa_points)
    
    # Check each placement: point is inside if R(-a)(w - (x,y)) is in base
    # R(-a) is rotation by -a degrees
    inside_all = np.ones(n_points, dtype=bool)
    
    for k in range(len(densified)):
        x, y, a = densified[k]
        if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(a):
            continue
        a_rad = np.radians(a)
        cos_a = np.cos(a_rad)
        sin_a = np.sin(a_rad)
        
        # w - (x, y)
        dx = sofa_points[:, 0] - x
        dy = sofa_points[:, 1] - y
        
        # R(-a) * (dx, dy): rotate by -a
        # R(-a) = [[cos(a), sin(a)], [-sin(a), cos(a)]]
        u = cos_a * dx + sin_a * dy
        v = -sin_a * dx + cos_a * dy
        
        # Check if in base corridor
        inside = _is_inside_base(u, v)
        inside_all &= inside
        
        # Early termination if no points remain
        if not inside_all.any():
            break
        
        # Deadline check
        if deadline is not None and time.time() > deadline:
            break
    
    count = inside_all.sum()
    return count * 0.0001


def moving_sofa_2d() -> np.ndarray:
    """
    Optimize the path using directed random walk with adaptive step sizes.
    """
    start_time = time.time()
    deadline = start_time + 580.0  # Safety margin before 630s limit
    
    # Initial path from the seed
    best_path = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ], dtype=np.float64)
    
    best_area = _compute_sofa_area(best_path, deadline)
    
    # Directed random walk with adaptive step sizes
    n_steps = 200
    step_size = 0.05
    improvement_threshold = 0.001
    
    for step in range(n_steps):
        if time.time() > deadline - 10.0:
            break
        
        # Generate a perturbation
        n_points = len(best_path)
        perturbation = np.random.randn(n_points, 3) * step_size
        
        # Ensure angles stay in reasonable range
        new_angles = best_path[:, 2] + perturbation[:, 2]
        # Keep angles monotonic-ish: first ~10-20, last ~80-90
        new_angles[0] = np.clip(new_angles[0], 5, 30)
        new_angles[-1] = np.clip(new_angles[-1], 70, 89)
        
        # Enforce monotonicity of angles
        for i in range(1, n_points):
            new_angles[i] = max(new_angles[i], new_angles[i-1] + 0.5)
            new_angles[i] = min(new_angles[i], 89.0)
        
        candidate = best_path.copy()
        candidate[:, 2] = new_angles
        candidate[:, 0] += perturbation[:, 0]
        candidate[:, 1] += perturbation[:, 1]
        
        # Keep x, y in reasonable bounds
        candidate[:, 0] = np.clip(candidate[:, 0], -1.5, 0.5)
        candidate[:, 1] = np.clip(candidate[:, 1], -0.5, 0.5)
        
        area = _compute_sofa_area(candidate, deadline)
        
        if area > best_area + improvement_threshold * 0.01:
            best_path = candidate
            best_area = area
            step_size = min(step_size * 1.2, 0.15)
        else:
            step_size = max(step_size * 0.85, 0.01)
    
    return best_path


# EVOLVE-BLOCK-END
