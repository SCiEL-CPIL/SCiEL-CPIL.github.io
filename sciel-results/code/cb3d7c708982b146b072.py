"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize

N = 13
DEADLINE = 138.0  # tighter safety margin for more reliable returns

# Pre-compute all triples for fast vectorized scoring
_triple_i = []
_triple_j = []
_triple_k = []
for _i in range(N):
    for _j in range(_i + 1, N):
        for _k in range(_j + 1, N):
            _triple_i.append(_i)
            _triple_j.append(_j)
            _triple_k.append(_k)
_triple_i = np.array(_triple_i, dtype=np.int32)
_triple_j = np.array(_triple_j, dtype=np.int32)
_triple_k = np.array(_triple_k, dtype=np.int32)


def _score_points(pts):
    """Compute min triangle area / convex hull area for given points."""
    n = len(pts)
    if n < 3:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)

    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0

    x = pts[:, 0]
    y = pts[:, 1]
    xi = x[_triple_i]
    xj = x[_triple_j]
    xk = x[_triple_k]
    yi = y[_triple_i]
    yj = y[_triple_j]
    yk = y[_triple_k]

    cross = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
    areas = 0.5 * cross
    min_area = np.min(areas)

    if min_area < 1e-15:
        return 0.0

    return min_area / hull_area


def _soft_min_score(params, beta=20.0):
    """
    Compute a smooth approximation of the min triangle area / hull area.
    Uses log-sum-exp of negative areas to create a differentiable proxy for min.
    """
    pts = params.reshape(N, 2)
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return -1e10
    except Exception:
        return -1e10

    x = pts[:, 0]
    y = pts[:, 1]
    xi = x[_triple_i]
    xj = x[_triple_j]
    xk = x[_triple_k]
    yi = y[_triple_i]
    yj = y[_triple_j]
    yk = y[_triple_k]

    cross = np.abs((xj - xi) * (yk - yi) - (yj - yi) * (xk - xi))
    areas = 0.5 * cross
    
    # Avoid log(0)
    areas = np.maximum(areas, 1e-12)
    
    # Soft min: -log(sum(exp(-beta * area))) / beta
    # We want to maximize the soft min, so we minimize the negative of it.
    # The evaluator scores based on hard min, so we use this only for optimization guidance.
    # Normalization by hull area is important to keep scale invariant.
    normalized_areas = areas / hull_area
    
    # Log-sum-exp trick for stability
    min_val = np.min(normalized_areas)
    exp_terms = np.exp(-beta * (normalized_areas - min_val))
    sum_exp = np.sum(exp_terms)
    
    # Soft min value
    soft_min = min_val - (1.0 / beta) * np.log(sum_exp)
    
    return -soft_min # Negate for minimization


def _neg_score(params):
    """Negative score for minimization (Hard Min)."""
    pts = params.reshape(N, 2)
    return -_score_points(pts)


def _regular_polygon(n, radius, center=(0.0, 0.0), angle_offset=0.0):
    if n <= 0:
        return np.array([]).reshape(0, 2)
    angles = np.linspace(angle_offset, 2 * np.pi + angle_offset, n, endpoint=False)
    xs = center[0] + radius * np.cos(angles)
    ys = center[1] + radius * np.sin(angles)
    return np.column_stack([xs, ys])


def _make_two_layer(n_hull, hull_radius, inner_radius, hull_offset=0.0, inner_offset=0.0):
    """Two-layer: n_hull polygon vertices + (N-n_hull) inner polygon vertices."""
    n_inner = N - n_hull
    if n_hull < 3 or n_inner < 0:
        return None
    if n_inner == 0:
        return _regular_polygon(n_hull, hull_radius, angle_offset=hull_offset)
    hull_pts = _regular_polygon(n_hull, hull_radius, angle_offset=hull_offset)
    inner_pts = _regular_polygon(n_inner, inner_radius, angle_offset=inner_offset)
    return np.vstack([hull_pts, inner_pts])


def construct():
    t_start = time.time()
    best_score = -float('inf')
    best_params = None

    def _update(pts):
        nonlocal best_score, best_params
        if pts is None or pts.shape != (N, 2):
            return
        s = _score_points(pts)
        if s > best_score:
            best_score = s
            best_params = pts.copy()

    # ============================================================
    # Phase 1: Symmetric Grid Search
    # Focus on two-layer configurations which are strong candidates
    # ============================================================
    # We will search over n_hull, hull_radius, inner_radius, offsets
    # Use a coarser grid first, then refine the best ones
    
    # Initial coarse grid
    hull_radii = np.linspace(0.5, 1.5, 15)
    inner_radii = np.linspace(0.0, 0.8, 15)
    offset_vals = np.linspace(0, np.pi / 2, 6)

    best_symmetric_params = None
    best_symmetric_score = -float('inf')

    for n_hull in range(3, N + 1):
        n_inner = N - n_hull
        if n_inner < 0:
            continue
        for hr in hull_radii:
            if time.time() - t_start > DEADLINE - 60:
                break
            if n_inner == 0:
                for ho in offset_vals:
                    cand = _make_two_layer(n_hull, hr, 0.0, hull_offset=ho)
                    s = _score_points(cand)
                    if s > best_symmetric_score:
                        best_symmetric_score = s
                        best_symmetric_params = cand.copy()
            else:
                for ir in inner_radii:
                    if time.time() - t_start > DEADLINE - 60:
                        break
                    for ho in offset_vals:
                        for io in offset_vals:
                            if time.time() - t_start > DEADLINE - 60:
                                break
                            cand = _make_two_layer(n_hull, hr, ir,
                                                   hull_offset=ho, inner_offset=io)
                            s = _score_points(cand)
                            if s > best_symmetric_score:
                                best_symmetric_score = s
                                best_symmetric_params = cand.copy()
        
        if time.time() - t_start > DEADLINE - 60:
            break

    if best_symmetric_params is not None:
        _update(best_symmetric_params)

    # ============================================================
    # Phase 2: Local Refinement using Soft-Min Objective
    # Start from best symmetric and regular polygon
    # ============================================================
    
    # Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x0 = np.column_stack([np.cos(angles), np.sin(angles)])
    try:
        hull0 = ConvexHull(x0)
        area0 = hull0.volume
        if area0 > 1e-12:
            x0 = x0 * np.sqrt(1.0 / area0)
    except Exception:
        pass
    _update(x0)

    starts = []
    if best_symmetric_params is not None:
        starts.append(best_symmetric_params.flatten().copy())
    starts.append(x0.flatten().copy())

    # Add some random perturbations of the best found so far
    rng = np.random.RandomState(42)
    if best_params is not None:
        for _ in range(5):
            pert = best_params.flatten() + rng.normal(0, 0.1, size=2 * N)
            starts.append(pert)

    # Optimize using soft-min to escape local minima
    for s in starts:
        if time.time() - t_start > DEADLINE - 20:
            break
        try:
            # Use soft-min for the main optimization step
            res = minimize(
                _soft_min_score,
                s,
                method='L-BFGS-B',
                bounds=[(-2.0, 2.0)] * (2 * N),
                options={'maxiter': 500, 'maxfun': 10000, 'ftol': 1e-12}
            )
            cand = res.x.reshape(N, 2)
            _update(cand)
            
            # Polish with hard min if time permits
            if time.time() - t_start < DEADLINE - 10:
                res_hard = minimize(
                    _neg_score,
                    cand.flatten(),
                    method='L-BFGS-B',
                    bounds=[(-2.0, 2.0)] * (2 * N),
                    options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-12}
                )
                cand_hard = res_hard.x.reshape(N, 2)
                _update(cand_hard)
        except Exception:
            pass

    # ============================================================
    # Phase 3: Differential Evolution with Soft-Min
    # Global search to find new basins of attraction
    # ============================================================
    if time.time() - t_start < DEADLINE - 30:
        bounds = [(-1.5, 1.5)] * (2 * N)
        try:
            result = differential_evolution(
                _soft_min_score,
                bounds,
                seed=42,
                maxiter=30,
                popsize=15,
                tol=1e-5,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False,
                init='sobol',
                updating='deferred',
                workers=1
            )
            if result.fun < 0: # Check if it found something valid
                candidate = result.x.reshape(N, 2)
                _update(candidate)
                
                # Polish the DE result
                if time.time() - t_start < DEADLINE - 10:
                    res_polish = minimize(
                        _neg_score,
                        candidate.flatten(),
                        method='L-BFGS-B',
                        bounds=[(-2.0, 2.0)] * (2 * N),
                        options={'maxiter': 300, 'maxfun': 6000, 'ftol': 1e-12}
                    )
                    cand_polish = res_polish.x.reshape(N, 2)
                    _update(cand_polish)
        except Exception:
            pass

    # ============================================================
    # Phase 4: Final Polish
    # ============================================================
    if best_params is not None and time.time() - t_start < DEADLINE - 5:
        try:
            res_nm = minimize(
                _neg_score,
                best_params.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12,
                         'adaptive': True}
            )
            cand = res_nm.x.reshape(N, 2)
            _update(cand)
        except Exception:
            pass

    if best_params is None:
        return x0
    return best_params
