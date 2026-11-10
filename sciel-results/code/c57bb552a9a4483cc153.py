# EVOLVE-BLOCK-START
import numpy as np
import time

def _point_in_base(u, v):
    """Check if point (u, v) is in the base L-shaped corridor.
    Base is union of {0<=u<=1, -3<=v<=1} and {-3<=u<=1, 0<=v<=1}.
    """
    return ((u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)) | \
           ((u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0))


def _evaluate_sofa_area(path):
    """Evaluate the sofa area for a given path.
    The evaluator prepends (0,0,0) and appends (-2,0,90).
    Then densifies with m = max(1, ceil(|a2-a1|)) steps.
    """
    # Build the full path with endpoints
    full_path = np.vstack([
        np.array([[0.0, 0.0, 0.0]]),
        path,
        np.array([[-2.0, 0.0, 90.0]])
    ])
    
    # Densify: between consecutive rows, add linear interpolations
    # m = max(1, ceil(|a2 - a1|)) equal steps
    densified = [full_path[0]]
    for i in range(len(full_path) - 1):
        p1 = full_path[i]
        p2 = full_path[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da)))
        # m steps means m+1 points including endpoints, but we skip the start
        # Actually, "m equal steps" means we divide the interval into m sub-intervals
        # So we get m+1 points, but the first is p1 which we already have
        for k in range(1, m + 1):
            t = k / m
            interp = p1 + t * (p2 - p1)
            densified.append(interp)
    
    densified = np.array(densified)
    
    # Sofa grid points
    xs = np.arange(-3.0, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    points = np.column_stack([X.ravel(), Y.ravel()])
    
    # For each placement, check which points are inside
    # A point w is inside placement (x, y, a) if R(-a)(w - (x, y)) lies in the base
    n_points = len(points)
    inside_all = np.ones(n_points, dtype=bool)
    
    for placement in densified:
        x, y, a_deg = placement
        a_rad = np.radians(a_deg)
        # R(-a) is rotation by -a
        cos_a = np.cos(-a_rad)
        sin_a = np.sin(-a_rad)
        
        # w - (x, y)
        dx = points[:, 0] - x
        dy = points[:, 1] - y
        
        # Apply R(-a)
        u = cos_a * dx - sin_a * dy
        v = sin_a * dx + cos_a * dy
        
        # Check if (u, v) is in base
        inside = _point_in_base(u, v)
        inside_all &= inside
        
        # Early exit if no points remain
        if not np.any(inside_all):
            break
    
    area = np.sum(inside_all) * 0.0001
    return area


def moving_sofa_2d() -> np.ndarray:
    """
    Find a path for moving a sofa through an L-shaped corridor.
    Returns an array of shape (n, 3) with rows (x, y, angle_degrees).
    """
    start_time = time.time()
    deadline = start_time + 570  # 570 seconds to leave margin before 630s kill
    
    # Start with the baseline path
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
    
    best_area = _evaluate_sofa_area(best_path)
    
    # Local optimization: perturb each point slightly and keep improvements
    # Use a simple hill-climbing with random restarts
    np.random.seed(42)
    
    # Define a set of perturbation scales
    scales = [0.1, 0.05, 0.02, 0.01]
    
    for scale in scales:
        for _ in range(200):
            if time.time() > deadline:
                break
            
            # Create a perturbed copy
            perturbed = best_path.copy()
            for i in range(len(perturbed)):
                # Perturb x, y, and angle
                perturbed[i, 0] += np.random.randn() * scale
                perturbed[i, 1] += np.random.randn() * scale
                perturbed[i, 2] += np.random.randn() * scale * 0.5  # angle less sensitive
                
                # Keep angle in reasonable range
                perturbed[i, 2] = np.clip(perturbed[i, 2], 5.0, 85.0)
            
            # Evaluate
            new_area = _evaluate_sofa_area(perturbed)
            
            if new_area > best_area:
                best_area = new_area
                best_path = perturbed
    
    # Try adding/removing points
    # Try adding a point in the middle
    if time.time() < deadline:
        for _ in range(100):
            if time.time() > deadline:
                break
            
            # Try inserting a new point
            insert_pos = np.random.randint(1, len(best_path))
            new_point = best_path[insert_pos - 1].copy()
            new_point += np.random.randn(3) * 0.05
            new_point[2] = np.clip(new_point[2], 5.0, 85.0)
            
            candidate = np.vstack([best_path[:insert_pos], new_point, best_path[insert_pos:]])
            new_area = _evaluate_sofa_area(candidate)
            
            if new_area > best_area:
                best_area = new_area
                best_path = candidate
    
    # Try removing a point
    if time.time() < deadline and len(best_path) > 2:
        for _ in range(100):
            if time.time() > deadline:
                break
            
            remove_pos = np.random.randint(1, len(best_path))
            candidate = np.delete(best_path, remove_pos, axis=0)
            new_area = _evaluate_sofa_area(candidate)
            
            if new_area > best_area:
                best_area = new_area
                best_path = candidate
    
    return best_path


# EVOLVE-BLOCK-END
