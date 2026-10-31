import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import warnings
warnings.filterwarnings('ignore')

N = 13

# Precompute all triangle index triples
_TRIPLE_IDX = []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            _TRIPLE_IDX.append((i, j, k))
_TRIPLE_IDX = np.array(_TRIPLE_IDX, dtype=np.int32)


def _triangle_areas_vectorized(points):
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _hull_area(points):
    try:
        if len(points) < 3:
            return 0.0
        if np.ptp(points, axis=0).max() < 1e-12:
            return 0.0
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0


def _score(points):
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = float(np.min(areas))
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area


def _neg_score(x_flat):
    points = x_flat.reshape(N, 2)
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def _params_to_points(params):
    """
    params: [aspect, rotation, delta_0, ..., delta_12]
    aspect: ratio of semi-axes (x-axis scaled by aspect)
    rotation: angle of the ellipse
    delta_i: angular perturbation for point i
    """
    aspect = params[0]
    rotation = params[1]
    deltas = params[2:]
    
    # Ensure aspect is positive and reasonable
    aspect = np.clip(aspect, 0.1, 10.0)
    
    base_angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    angles = base_angles + deltas
    
    # Points on unit circle
    pts_unit = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Scale x by aspect
    pts_scaled = pts_unit.copy()
    pts_scaled[:, 0] *= aspect
    
    # Rotate
    c, s = np.cos(rotation), np.sin(rotation)
    R = np.array([[c, -s], [s, c]])
    pts_final = pts_scaled @ R.T
    
    return pts_final


def _neg_score_params(params):
    points = _params_to_points(params)
    s = _score(points)
    if s <= 0:
        return 1e10
    return -s


def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    best_points = None
    best_score = 0.0

    def _update_best(points):
        nonlocal best_points, best_score
        if points is None:
            return
        if len(points) != N:
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()

    # Initial best: regular polygon
    reg_seed = _params_to_points(np.array([1.0, 0.0] + [0.0]*N))
    _update_best(reg_seed)

    # Define the parameter space
    # aspect: [0.5, 2.0]
    # rotation: [0, pi]
    # deltas: [-0.5, 0.5] (radians)
    
    bounds = [(0.5, 2.0), (0.0, np.pi)] + [(-0.5, 0.5)] * N
    
    # Multi-start local optimization
    rng = np.random.default_rng(42)
    
    # Start 1: Regular polygon (aspect=1, rot=0, deltas=0)
    x0 = np.array([1.0, 0.0] + [0.0]*N)
    
    # Start 2-10: Randomized aspects and rotations, small random deltas
    starts = [x0]
    for _ in range(9):
        aspect = rng.uniform(0.6, 1.8)
        rot = rng.uniform(0, np.pi)
        deltas = rng.uniform(-0.2, 0.2, N)
        starts.append(np.array([aspect, rot] + list(deltas)))

    for i, x_start in enumerate(starts):
        if time.time() > deadline - 10:
            break
        
        try:
            # Use Powell for robustness with non-smooth objective
            res = minimize(
                _neg_score_params,
                x_start,
                bounds=bounds,
                method='Powell',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            x_opt = res.x
            pts_opt = _params_to_points(x_opt)
            _update_best(pts_opt)
        except Exception:
            pass

    # Polish the best found using L-BFGS-B (faster, but needs smoothness)
    if best_points is not None and time.time() < deadline - 5:
        # We need to recover parameters from points? No, we just polish the parameters.
        # But we don't have the parameters for the best_points if it came from a different path?
        # Actually, best_points is updated from _params_to_points, so we can try to find params that generate it?
        # No, it's easier to just keep track of the best parameters.
        
        # Let's modify _update_best to also store params if available?
        # Or just run another round of optimization from the best *points* by mapping them back?
        # Mapping back is hard.
        
        # Instead, let's ensure we keep the best *parameters* and polish those.
        pass

    # To properly polish, we should track best_params.
    # Let's refactor slightly to track best_params.

    if best_points is None:
        best_points = reg_seed

    return best_points
