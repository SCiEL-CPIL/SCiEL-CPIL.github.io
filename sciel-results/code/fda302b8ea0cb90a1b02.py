import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
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
    """Vectorized minimum triangle area for all triples."""
    p1 = points[_TRIPLE_I]
    p2 = points[_TRIPLE_J]
    p3 = points[_TRIPLE_K]
    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return np.min(0.5 * np.abs(cross))

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

def _make_concentric_rings(n_rings, ring_sizes, radii, angle_offsets):
    """
    Create points in concentric rings.
    n_rings: number of rings
    ring_sizes: list of ints, number of points in each ring (must sum to N)
    radii: list of floats, radius of each ring
    angle_offsets: list of floats, angular offset for each ring
    """
    points = []
    for i in range(n_rings):
        n_pts = ring_sizes[i]
        r = radii[i]
        offset = angle_offsets[i]
        if n_pts == 1:
            angles = np.array([0.0])
        else:
            angles = np.linspace(0, 2*np.pi, n_pts, endpoint=False) + offset
        x = r * np.cos(angles)
        y = r * np.sin(angles)
        points.append(np.column_stack((x, y)))
    return np.vstack(points)

def _optimize_rings(ring_sizes, initial_radii, initial_offsets, deadline):
    """
    Optimize radii and angle offsets for a given ring structure.
    Returns (best_points, best_score).
    """
    n_rings = len(ring_sizes)
    
    def _eval_ring_params(params):
        radii = params[:n_rings]
        offsets = params[n_rings:]
        # Ensure radii are positive and sorted (inner to outer)
        radii = np.maximum(radii, 0.01)
        radii = np.sort(radii)
        # Normalize so outermost ring is at radius 1
        max_r = radii[-1]
        if max_r < 1e-10:
            return 0.0
        radii = radii / max_r
        points = _make_concentric_rings(n_rings, ring_sizes, radii, offsets)
        return get_score(points)
    
    # Initial point
    init_params = np.concatenate([initial_radii, initial_offsets])
    best_params = init_params.copy()
    best_score = _eval_ring_params(init_params)
    
    # Use differential evolution for global search
    bounds = [(0.01, 1.0)] * n_rings + [(0, 2*np.pi)] * n_rings
    
    try:
        if time.time() < deadline - 20.0:
            de_result = differential_evolution(
                lambda p: -_eval_ring_params(p),
                bounds,
                seed=42,
                maxiter=50,
                tol=1e-6,
                popsize=12,
                workers=-1
            )
            de_score = -de_result.fun
            if de_score > best_score:
                best_score = de_score
                best_params = de_result.x
    except Exception:
        pass
    
    # Local refinement with Powell
    try:
        if time.time() < deadline - 10.0:
            result = minimize(
                lambda p: -_eval_ring_params(p),
                best_params,
                method='Powell',
                options={'maxiter': 100, 'xtol': 1e-8, 'ftol': 1e-8}
            )
            powell_score = -result.fun
            if powell_score > best_score:
                best_score = powell_score
                best_params = result.x
    except Exception:
        pass
    
    # Get final points
    radii = np.maximum(best_params[:n_rings], 0.01)
    radii = np.sort(radii)
    max_r = radii[-1]
    if max_r < 1e-10:
        max_r = 1.0
    radii = radii / max_r
    offsets = best_params[n_rings:]
    points = _make_concentric_rings(n_rings, ring_sizes, radii, offsets)
    
    return points, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    # 1. Initialize with regular polygon
    regular_angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    regular_points = np.column_stack((np.cos(regular_angles), np.sin(regular_angles)))
    regular_score = get_score(regular_points)
    best_score = regular_score
    best_points = regular_points.copy()
    
    # 2. Try different concentric ring structures
    # For N=14, possible ring structures (summing to 14):
    # Single ring: [14]
    # Two rings: [7,7], [8,6], [9,5], [10,4], [11,3], [12,2], [13,1]
    # Three rings: [5,5,4], [6,5,3], [6,4,4], [7,4,3], [7,5,2], [7,6,1], [8,4,2], [8,5,1], [9,3,2], [9,4,1], [10,2,2], [10,3,1], [11,2,1], [12,1,1]
    # Four rings: [4,4,3,3], [5,3,3,3], [5,4,2,3], [5,5,2,2], [6,3,2,3], [6,4,2,2], [6,5,1,2], [7,2,2,3], [7,3,2,2], [7,4,1,2], [8,2,2,2], [8,3,1,2], [9,2,1,2], [10,1,1,2], [11,1,1,1]
    
    ring_structures = [
        [14],
        [7, 7],
        [8, 6],
        [9, 5],
        [10, 4],
        [11, 3],
        [12, 2],
        [13, 1],
        [5, 5, 4],
        [6, 5, 3],
        [6, 4, 4],
        [7, 4, 3],
        [7, 5, 2],
        [7, 6, 1],
        [8, 4, 2],
        [8, 5, 1],
        [9, 3, 2],
        [9, 4, 1],
        [10, 2, 2],
        [10, 3, 1],
        [11, 2, 1],
        [12, 1, 1],
        [4, 4, 3, 3],
        [5, 3, 3, 3],
        [5, 4, 2, 3],
        [5, 5, 2, 2],
        [6, 3, 2, 3],
        [6, 4, 2, 2],
        [6, 5, 1, 2],
        [7, 2, 2, 3],
        [7, 3, 2, 2],
        [7, 4, 1, 2],
        [8, 2, 2, 2],
        [8, 3, 1, 2],
        [9, 2, 1, 2],
        [10, 1, 1, 2],
        [11, 1, 1, 1],
    ]
    
    for ring_sizes in ring_structures:
        if time.time() > deadline - 30.0:
            break
        
        n_rings = len(ring_sizes)
        
        # Initial radii: evenly spaced from 0.2 to 1.0
        if n_rings == 1:
            initial_radii = [1.0]
        else:
            initial_radii = np.linspace(0.3, 1.0, n_rings).tolist()
        
        # Initial offsets: 0 for all rings
        initial_offsets = [0.0] * n_rings
        
        try:
            points, score = _optimize_rings(ring_sizes, initial_radii, initial_offsets, deadline)
            if score > best_score:
                best_score = score
                best_points = points.copy()
        except Exception:
            pass
    
    # 3. Also try 2D local refinement from best result
    if best_points is not None and time.time() < deadline - 15.0:
        try:
            result = minimize(
                lambda x: -get_score(x.reshape(N, 2)),
                best_points.flatten(),
                method='Powell',
                options={'maxiter': 150, 'xtol': 1e-8, 'ftol': 1e-8}
            )
            if result.fun < 0:
                score = -result.fun
                if score > best_score:
                    best_score = score
                    best_points = result.x.reshape(N, 2).copy()
        except Exception:
            pass
    
    # 4. Final check and return
    if best_points is None:
        best_points = regular_points

    return best_points
