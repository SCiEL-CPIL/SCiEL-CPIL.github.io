import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 14

# Precompute triple indices for vectorized area calculation
_i_list, _j_list, _k_list = [], [], []
for _i in range(N):
    for _j in range(_i + 1, N):
        for _k in range(_j + 1, N):
            _i_list.append(_i)
            _j_list.append(_j)
            _k_list.append(_k)
_TRIPLE_I = np.array(_i_list)
_TRIPLE_J = np.array(_j_list)
_TRIPLE_K = np.array(_k_list)

def _compute_min_area(points):
    """Compute the minimum triangle area for a set of points."""
    p1 = points[_TRIPLE_I]
    p2 = points[_TRIPLE_J]
    p3 = points[_TRIPLE_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    return np.min(areas)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def get_score(points):
    """Score = min triangle area / convex hull area."""
    ha = _hull_area(points)
    if ha <= 1e-12:
        return 0.0
    ma = _compute_min_area(points)
    if ma < 1e-15:
        return 0.0
    return ma / ha

def objective_angles(angles):
    """
    Objective for optimization: minimize negative min area.
    Assumes points are on the unit circle, so hull area is constant (approx).
    Actually, we maximize min area.
    """
    x = np.cos(angles)
    y = np.sin(angles)
    points = np.column_stack((x, y))
    # For points on unit circle, hull area is constant if they are in convex position
    # We just need to maximize min area
    ma = _compute_min_area(points)
    return -ma

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_score = 0.0
    best_points = None
    
    # Initial guess: Regular polygon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    
    # Multi-start optimization
    # We optimize the angles directly.
    # Powell's method is robust for this non-smooth objective.
    
    rng = np.random.default_rng(42)
    
    # Try a few perturbations
    num_restarts = 5
    
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
            
        if i == 0:
            current_angles = angles.copy()
        else:
            # Small random perturbation
            current_angles = angles + rng.normal(0, 0.1, N)
            # Sort to maintain order (optional, but helps with local minima)
            current_angles = np.sort(current_angles)
        
        try:
            result = minimize(
                objective_angles, 
                current_angles, 
                method='Powell', 
                options={'maxiter': 200, 'xtol': 1e-8, 'ftol': 1e-8}
            )
            
            opt_angles = result.x
            # Ensure angles are within [0, 2pi) and sorted
            opt_angles = np.sort(opt_angles % (2 * np.pi))
            
            x = np.cos(opt_angles)
            y = np.sin(opt_angles)
            points = np.column_stack((x, y))
            
            score = get_score(points)
            
            if score > best_score:
                best_score = score
                best_points = points.copy()
                
        except Exception:
            pass

    if best_points is None:
        # Fallback to regular polygon
        x = np.cos(angles)
        y = np.sin(angles)
        best_points = np.column_stack((x, y))

    return best_points
