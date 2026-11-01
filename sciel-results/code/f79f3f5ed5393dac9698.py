import numpy as np
import scipy.spatial
import scipy.optimize
import time
import itertools

N = 14
N_OUTER = 7
N_INNER = 7

def _get_triangle_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

_TRI_INDICES = _get_triangle_indices(N)
_I = _TRI_INDICES[:, 0]
_J = _TRI_INDICES[:, 1]
_K = _TRI_INDICES[:, 2]

def _min_triangle_area(points):
    """Compute minimum triangle area for given points."""
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = scipy.spatial.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Score = min triangle area / convex hull area."""
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        mta = _min_triangle_area(points)
        return mta / ha
    except Exception:
        return 0.0

def _polar_to_cartesian(angles_outer, angles_inner, r_ratio, R=1.0):
    """
    Convert polar parameters to Cartesian points.
    Points 0..6 are on outer circle (radius R).
    Points 7..13 are on inner circle (radius r_ratio * R).
    """
    outer_x = R * np.cos(angles_outer)
    outer_y = R * np.sin(angles_outer)
    inner_r = r_ratio * R
    inner_x = inner_r * np.cos(angles_inner)
    inner_y = inner_r * np.sin(angles_inner)
    
    x = np.concatenate([outer_x, inner_x])
    y = np.concatenate([outer_y, inner_y])
    return np.column_stack([x, y])

def _neg_score_polar(params):
    """
    Objective function for polar coordinate optimization.
    params: [angles_outer (7), angles_inner (7), r_ratio (1)]
    """
    angles_outer = params[0:7]
    angles_inner = params[7:14]
    r_ratio = params[14]
    
    # Constrain r_ratio to be between 0.1 and 0.9 to avoid degenerate cases
    r_ratio = np.clip(r_ratio, 0.1, 0.9)
    
    points = _polar_to_cartesian(angles_outer, angles_inner, r_ratio, R=1.0)
    
    # The score is scale invariant, so we can compute it on the unit circle hull
    # and it will be the same as normalized.
    # However, to be safe and consistent with the evaluator:
    return -_score(points)

def _regular_polar_init(seed=0):
    """
    Initialize with regular polygons on two circles.
    """
    rng = np.random.RandomState(seed)
    # Regular 7-gon on outer
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False) + rng.uniform(0, 2*np.pi)
    # Regular 7-gon on inner, possibly rotated
    angles_inner = np.linspace(0, 2 * np.pi, 7, endpoint=False) + rng.uniform(0, 2*np.pi)
    # Initial radius ratio
    r_ratio = rng.uniform(0.4, 0.7)
    
    params = np.concatenate([angles_outer, angles_inner, [r_ratio]])
    return params

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_score = 0.0
    best_points = None
    
    def _try_optimize(init_params, method='L-BFGS-B'):
        nonlocal best_score, best_points
        if time.time() > deadline:
            return None
        try:
            # Initial check
            init_pts = _polar_to_cartesian(init_params[0:7], init_params[7:14], init_params[14])
            init_score = _score(init_pts)
            
            res = scipy.optimize.minimize(
                _neg_score_polar,
                init_params,
                method=method,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            
            opt_params = res.x
            opt_pts = _polar_to_cartesian(opt_params[0:7], opt_params[7:14], opt_params[14])
            sc = _score(opt_pts)
            
            if sc > best_score:
                best_score = sc
                best_points = opt_pts.copy()
            
            return (sc, opt_pts)
        except Exception as e:
            return None

    # Generate diverse initializations
    inits = []
    
    # 1. Regular symmetric starts
    for seed in range(20):
        inits.append(_regular_polar_init(seed))
        
    # 2. Random starts
    rng = np.random.RandomState(42)
    for _ in range(20):
        angles_outer = rng.uniform(0, 2*np.pi, 7)
        angles_inner = rng.uniform(0, 2*np.pi, 7)
        r_ratio = rng.uniform(0.2, 0.8)
        inits.append(np.concatenate([angles_outer, angles_inner, [r_ratio]]))
        
    # 3. Specific known good configurations (e.g., near 0.5 ratio)
    for ratio in [0.3, 0.5, 0.6, 0.7]:
        angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        angles_inner = np.linspace(np.pi/7, 2 * np.pi + np.pi/7, 7, endpoint=False) # Interleaved
        inits.append(np.concatenate([angles_outer, angles_inner, [ratio]]))

    # Optimize from each initialization
    for i, init in enumerate(inits):
        if time.time() > deadline:
            break
            
        # Use L-BFGS-B as it handles the smooth landscape well
        _try_optimize(init, 'L-BFGS-B')
        
        # Occasionally try Powell for robustness
        if i % 5 == 0 and time.time() < deadline - 10:
             _try_optimize(init, 'Powell')

    # If we still have time, do a few more random restarts
    while time.time() < deadline - 5:
        seed = int(time.time() * 1000) % 100000
        init = _regular_polar_init(seed)
        _try_optimize(init, 'L-BFGS-B')

    # Final fallback
    if best_points is None:
        # Fallback to regular 14-gon if optimization failed
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
        
    # Normalize the final points to unit hull area
    ha = _hull_area(best_points)
    if ha > 1e-12:
        best_points = best_points / np.sqrt(ha)
        
    return best_points
