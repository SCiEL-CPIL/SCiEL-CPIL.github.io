import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize
import time
import itertools

N = 14

# Precompute triple indices for N=14 to speed up area calculation
TRIPLE_INDICES = np.array([
    (i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)
])

def _calculate_score(points):
    """
    Calculate the score: min triangle area / convex hull area.
    points: (N, 2) array
    """
    if len(points) < 3:
        return 0.0
    
    # Calculate Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    # Calculate Minimum Triangle Area
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]

    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])

    min_area = 0.5 * np.min(np.abs(cross))

    return min_area / hull_area

def _build_points_from_params(params):
    """
    Build 14 points from 14 parameters:
    - 7 radial offsets (one per heptagon vertex pair)
    - 3 angular offsets (for the 3 pairs of points per vertex)
    - 3 affine parameters (rotation, scale_x, scale_y/shear)
    """
    radials = params[:7]
    angles = params[7:10]
    affine = params[10:13]
    # 14th param: overall rotation
    
    # Base heptagon vertices
    base_angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    
    # Each vertex has a pair of points offset radially
    # Point 2*i and 2*i+1 are the pair for vertex i
    points = np.zeros((14, 2))
    
    for i in range(7):
        r = radials[i]
        # Two points near vertex i, separated by small angular offset
        a1 = base_angles[i] + angles[0] * 0.1
        a2 = base_angles[i] - angles[0] * 0.1
        
        # Radial positions: one at r-0.05, one at r+0.05 (scaled by angles[1])
        r1 = r - 0.05 * angles[1]
        r2 = r + 0.05 * angles[1]
        
        points[2*i, 0] = r1 * np.cos(a1)
        points[2*i, 1] = r1 * np.sin(a1)
        points[2*i+1, 0] = r2 * np.cos(a2)
        points[2*i+1, 1] = r2 * np.sin(a2)
    
    # Apply affine transformation
    # Rotation by affine[0], scaling by (1+affine[1], 1+affine[2])
    theta = affine[0]
    sx = 1.0 + affine[1]
    sy = 1.0 + affine[2]
    
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta), np.cos(theta)]])
    S = np.array([[sx, 0], [0, sy]])
    
    M = S @ R
    points = points @ M.T
    
    return points

def _score_params(params):
    """
    Score function for parameter optimization.
    """
    points = _build_points_from_params(params)
    return _calculate_score(points)

def _soft_min_score_from_points(points, beta=100.0):
    """
    Soft-min surrogate for the minimum triangle area.
    """
    p_i = points[TRIPLE_INDICES[:, 0]]
    p_j = points[TRIPLE_INDICES[:, 1]]
    p_k = points[TRIPLE_INDICES[:, 2]]

    cross = (p_j[:, 0] - p_i[:, 0]) * (p_k[:, 1] - p_i[:, 1]) - \
            (p_j[:, 1] - p_i[:, 1]) * (p_k[:, 0] - p_i[:, 0])

    areas = 0.5 * np.abs(cross)

    # Soft-min: -1/beta * log(sum(exp(-beta * areas)))
    max_area = np.max(areas)
    shifted = -beta * (areas - max_area)
    log_sum = np.log(np.sum(np.exp(shifted))) + beta * max_area
    soft_min = max_area - log_sum / beta

    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-9:
            return 0.0
    except Exception:
        return 0.0

    return soft_min / hull_area

def _soft_score_params(params, beta=100.0):
    """
    Soft-min surrogate objective for gradient-based optimization.
    """
    points = _build_points_from_params(params)
    return _soft_min_score_from_points(points, beta)

def _optimize_pair_heptagon(deadline):
    """
    Optimize the pair-heptagon configuration with affine transform.
    """
    start_time = time.time()
    best_score = 0.0
    best_params = None

    # Parameter bounds
    # 7 radial offsets: [0.5, 1.0]
    # 3 angular offsets: [0.1, 1.0] (scaled internally)
    # 3 affine: rotation [-pi, pi], scale_x [-0.2, 0.2], scale_y [-0.2, 0.2]
    bounds = [(0.5, 1.0)] * 7 + [(0.1, 1.0)] * 3 + [(-np.pi, np.pi), (-0.2, 0.2), (-0.2, 0.2)]
    
    # Initial guess: regular heptagon with pairs close together
    init = np.array([0.75]*7 + [0.5, 0.5, 0.5] + [0.0, 0.0, 0.0])

    # Phase 1: Coarse grid search on key parameters
    if time.time() < deadline - 30:
        # Grid search on radial mean and angular separation
        radial_means = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
        ang_seps = [0.3, 0.5, 0.7, 1.0]
        
        for rm in radial_means:
            if time.time() > deadline - 30:
                break
            for as_ in ang_seps:
                if time.time() > deadline - 30:
                    break
                params = np.array([rm]*7 + [as_, 0.5, 0.5] + [0.0, 0.0, 0.0])
                score = _score_params(params)
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
    
    # Phase 2: Differential Evolution
    if time.time() < deadline - 20 and best_params is not None:
        try:
            res = differential_evolution(
                lambda p: -_score_params(p),
                bounds,
                x0=best_params,
                seed=42,
                maxiter=20,
                popsize=12,
                tol=1e-5,
                polish=False
            )
            if -res.fun > best_score:
                best_score = -res.fun
                best_params = res.x
        except Exception:
            pass
    
    # Phase 3: Gradient-based polish with Soft-Min Surrogate
    if time.time() < deadline - 10 and best_params is not None:
        try:
            current_params = best_params.copy()
            betas = [50.0, 100.0, 200.0, 500.0]
            
            for beta in betas:
                if time.time() > deadline - 10:
                    break
                res_soft = minimize(
                    lambda p: -_soft_score_params(p, beta),
                    current_params,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'ftol': 1e-12, 'gtol': 1e-10, 'maxiter': 300}
                )
                if -res_soft.fun > best_score:
                    best_score = -res_soft.fun
                    best_params = res_soft.x
                current_params = res_soft.x
        except Exception:
            pass
    
    # Phase 4: Final exact evaluation and Nelder-Mead polish
    if time.time() < deadline - 3 and best_params is not None:
        try:
            res_nm = minimize(
                lambda p: -_score_params(p),
                best_params,
                method='Nelder-Mead',
                options={'xatol': 1e-10, 'fatol': 1e-10, 'maxiter': 500}
            )
            if -res_nm.fun > best_score:
                best_score = -res_nm.fun
                best_params = res_nm.x
        except Exception:
            pass

    if best_params is None:
        return 0.0, None

    return best_score, best_params

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    best_overall_score = 0.0
    best_overall_points = None

    # Strategy: Pair-heptagon with affine transform
    score, params = _optimize_pair_heptagon(deadline)

    if score > best_overall_score:
        best_overall_score = score
        if params is not None:
            best_overall_points = _build_points_from_params(params)

    # Fallback if no valid points found
    if best_overall_points is None:
        # Simple regular heptagon with pairs
        params = np.array([0.75]*7 + [0.5, 0.5, 0.5] + [0.0, 0.0, 0.0])
        best_overall_points = _build_points_from_params(params)

    return best_overall_points
