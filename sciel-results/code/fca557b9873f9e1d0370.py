import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

N = 14
START_TIME = time.time()
DEADLINE = START_TIME + 115.0

def _get_triple_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=int)

TRIPLE_INDICES = _get_triple_indices(N)
I_IDX = TRIPLE_INDICES[:, 0]
J_IDX = TRIPLE_INDICES[:, 1]
K_IDX = TRIPLE_INDICES[:, 2]

def _score(points):
    """Compute min triangle area / convex hull area."""
    if len(points) < 3:
        return 0.0
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    if hull_area < 1e-12:
        return 0.0
    p_i = points[I_IDX]
    p_j = points[J_IDX]
    p_k = points[K_IDX]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    min_area = np.min(areas)
    return min_area / hull_area

def _points_from_params(params):
    """
    params = [r, theta_offset, global_rotation]
    r: inner ring radius ratio (0 < r < 1)
    theta_offset: angular offset of inner ring relative to outer
    global_rotation: overall rotation
    Outer ring: 7 points on unit circle
    Inner ring: 7 points on circle of radius r, rotated by theta_offset
    """
    r = params[0]
    theta = params[1]
    rot = params[2]
    
    n_outer = 7
    n_inner = 7
    
    # Outer regular heptagon
    angles_outer = np.linspace(0, 2*np.pi, n_outer, endpoint=False) + rot
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    
    # Inner regular heptagon
    angles_inner = np.linspace(0, 2*np.pi, n_inner, endpoint=False) + theta + rot
    inner_pts = r * np.column_stack([np.cos(angles_inner), np.sin(angles_inner)])
    
    return np.vstack([outer_pts, inner_pts])

def _neg_score_params(params):
    pts = _points_from_params(params)
    return -_score(pts)

def _score_params(params):
    pts = _points_from_params(params)
    return _score(pts)

def _optimize_full_points(flat_points):
    """Optimize all 28 coordinates starting from flat_points."""
    def neg_score(flat):
        pts = flat.reshape((N, 2))
        return -_score(pts)
    try:
        res = minimize(
            neg_score,
            flat_points,
            method='Nelder-Mead',
            options={'maxiter': 800, 'maxfev': 2000, 'xatol': 1e-9, 'fatol': 1e-11}
        )
        return res.x.reshape((N, 2)), -res.fun
    except Exception:
        return flat_points.reshape((N, 2)), _score(flat_points.reshape((N, 2)))

def construct():
    global START_TIME, DEADLINE
    START_TIME = time.time()
    DEADLINE = START_TIME + 115.0
    
    np.random.seed(42)
    
    best_score = -1.0
    best_points = None
    
    def _update_best(pts, score):
        nonlocal best_score, best_points
        if score > best_score:
            best_score = score
            best_points = pts.copy()
    
    # === Phase 1: Global optimization over 3 parameters (r, theta, rot) ===
    # Bounds: r in (0.05, 0.95), theta in [0, 2*pi/7) (symmetry), rot in [0, 2*pi/7)
    bounds = [(0.05, 0.95), (0, 2*np.pi/7), (0, 2*np.pi/7)]
    
    if time.time() < DEADLINE - 60:
        try:
            de_res = differential_evolution(
                _neg_score_params,
                bounds,
                seed=42,
                maxiter=80,
                popsize=15,
                tol=1e-8,
                atol=1e-10,
                init='sobol',
                polish=False,
                update_interval=1,
                workers=1
            )
            de_params = de_res.x
            de_score = _score_params(de_params)
            de_pts = _points_from_params(de_params)
            _update_best(de_pts, de_score)
        except Exception:
            pass
    
    # === Phase 2: Local refinement of the 3-parameter solution ===
    if best_points is not None and time.time() < DEADLINE - 30:
        # Find best 3-param starting points from a grid
        grid_candidates = []
        for r in np.linspace(0.2, 0.9, 15):
            for theta in np.linspace(0, 2*np.pi/7, 8, endpoint=False):
                for rot in [0.0, np.pi/7]:
                    grid_candidates.append([r, theta, rot])
        
        # Score grid
        grid_scores = []
        for gc in grid_candidates:
            s = _score_params(gc)
            grid_scores.append((s, gc))
        grid_scores.sort(key=lambda x: -x[0])
        
        # Local optimize top grid candidates in 3D parameter space
        for s, gc in grid_scores[:5]:
            if time.time() > DEADLINE - 25:
                break
            try:
                res = minimize(
                    _neg_score_params,
                    gc,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'maxfev': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                opt_pts = _points_from_params(res.x)
                opt_score = _score(opt_pts)
                _update_best(opt_pts, opt_score)
            except Exception:
                pass
    
    # === Phase 3: Full 28D local optimization from best 3-param solution ===
    if best_points is not None and time.time() < DEADLINE - 15:
        full_pts, full_score = _optimize_full_points(best_points.flatten())
        _update_best(full_pts, full_score)
    
    # === Phase 4: Try a few random full-space starts ===
    if time.time() < DEADLINE - 10:
        for trial in range(3):
            if time.time() > DEADLINE - 8:
                break
            # Random perturbation of best
            if best_points is not None:
                perturbed = best_points + np.random.randn(N, 2) * 0.1
            else:
                angles = np.linspace(0, 2*np.pi, N, endpoint=False)
                perturbed = np.column_stack([np.cos(angles), np.sin(angles)]) + np.random.randn(N, 2) * 0.3
            full_pts, full_score = _optimize_full_points(perturbed.flatten())
            _update_best(full_pts, full_score)
    
    # === Phase 5: Final polish ===
    if best_points is not None and time.time() < DEADLINE - 5:
        try:
            res = minimize(
                lambda flat: -_score(flat.reshape((N, 2))),
                best_points.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 500, 'ftol': 1e-14}
            )
            final_pts = res.x.reshape((N, 2))
            final_score = _score(final_pts)
            _update_best(final_pts, final_score)
        except Exception:
            pass
    
    if best_points is None:
        # Fallback: regular 14-gon
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_points
