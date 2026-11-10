import numpy as np
import time

def _compute_score(path):
    """Compute the raw score for a given path (without endpoints)."""
    # Prepend start, append end
    start = np.array([0.0, 0.0, 0.0])
    end = np.array([-2.0, 0.0, 90.0])
    
    # Build full path with interpolation
    full_path = [start]
    for i in range(len(path)):
        p1 = full_path[-1]
        p2 = path[i]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da)))
        for k in range(1, m + 1):
            t = k / m
            interp = p1 + t * (p2 - p1)
            full_path.append(interp)
    full_path.append(end)
    
    full_path = np.array(full_path)
    
    # Sofa grid
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    gx, gy = np.meshgrid(xs, ys, indexing='ij')
    points = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (N, 2)
    N = len(points)
    
    # For each placement, check if all points are inside the corridor
    # A point w is inside placement (x, y, a) if R(-a)(w - (x,y)) is in base corridor
    # Base corridor: {0<=u<=1, -3<=v<=1} OR {-3<=u<=1, 0<=v<=1}
    
    inside_all = np.ones(N, dtype=bool)
    
    for (cx, cy, a_deg) in full_path:
        a_rad = np.deg2rad(a_deg)
        ca, sa = np.cos(a_rad), np.sin(a_rad)
        # R(-a) = [[cos(a), sin(a)], [-sin(a), cos(a)]]
        # w - (cx, cy)
        dx = points[:, 0] - cx
        dy = points[:, 1] - cy
        # Rotate by -a
        u = ca * dx + sa * dy
        v = -sa * dx + ca * dy
        
        # Check if (u, v) is in base corridor
        in_arm1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        in_arm2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        in_corridor = in_arm1 | in_arm2
        
        inside_all &= in_corridor
        if not inside_all.any():
            break
    
    count = np.sum(inside_all)
    return count * 0.0001


def moving_sofa_2d() -> np.ndarray:
    """
    Find a path for moving a sofa through an L-shaped corridor.
    Returns: np.ndarray of shape (n, 3), n >= 1, all finite.
    """
    start_time = time.time()
    deadline = 550.0  # safety margin before 600s limit
    
    # Initial path - a reasonable starting point
    path = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ], dtype=np.float64)
    
    best_score = _compute_score(path)
    best_path = path.copy()
    
    # Local optimization using coordinate descent with adaptive step sizes
    n_params = len(path) * 3
    params = path.ravel().copy()
    step_sizes = np.array([0.05, 0.05, 2.0]) * np.ones(n_params)
    
    # Group parameters: each row has (x, y, angle)
    # x, y ranges roughly [-1.5, 0.5], angle [0, 90]
    
    max_iter = 2000
    for iteration in range(max_iter):
        if time.time() - start_time > deadline:
            break
        
        improved = False
        for i in range(n_params):
            if time.time() - start_time > deadline:
                    break
            
            orig = params[i]
            step = step_sizes[i]
            
            # Try positive step
            params[i] = orig + step
            candidate = params.reshape(len(path), 3)
            # Ensure angles are in reasonable range
            if candidate[:, 2].min() < -5 or candidate[:, 2].max() > 95:
                params[i] = orig
                continue
            score = _compute_score(candidate)
            if score > best_score:
                best_score = score
                best_path = candidate.copy()
                improved = True
                step_sizes[i] *= 1.2
            else:
                params[i] = orig
                step_sizes[i] *= 0.7
            
            # Try negative step
            params[i] = orig - step
            candidate = params.reshape(len(path), 3)
            if candidate[:, 2].min() < -5 or candidate[:, 2].max() > 95:
                params[i] = orig
                continue
            score = _compute_score(candidate)
            if score > best_score:
                best_score = score
                best_path = candidate.copy()
                improved = True
                step_sizes[i] *= 1.2
            else:
                params[i] = orig
                step_sizes[i] *= 0.7
        
        # If no improvement in a full pass, reduce all step sizes
        if not improved:
            step_sizes *= 0.9
            if step_sizes.max() < 0.001:
                break
    
    return best_path
