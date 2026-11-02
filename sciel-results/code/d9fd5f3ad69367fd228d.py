"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

N = 13
DEADLINE = 145.0  # safety margin before 150s kill

# Pre-compute all C(13,3)=286 triple indices for fast vectorized scoring
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
_num_triples = len(_triple_i)

# Symmetry-constrained configuration definitions
# Each config: (n_outer, n_middle, n_inner, r_outer, r_middle, r_inner, phase_outer, phase_middle, phase_inner)
# Total points must equal N=13
_SYMMETRY_CONFIGS = [
    # (n1, n2, n3, r1, r2, r3, p1, p2, p3)
    (9, 4, 0, 1.0, 0.5, 0.0, 0.0, 0.0, 0.0),    # 9+4
    (8, 5, 0, 1.0, 0.45, 0.0, 0.0, 0.0, 0.0),   # 8+5
    (7, 6, 0, 1.0, 0.55, 0.0, 0.0, 0.0, 0.0),   # 7+6
    (6, 5, 2, 1.0, 0.6, 0.3, 0.0, 0.0, 0.0),    # 6+5+2
    (5, 4, 4, 1.0, 0.6, 0.3, 0.0, 0.0, 0.0),    # 5+4+4
    (6, 4, 3, 1.0, 0.55, 0.25, 0.0, 0.0, 0.0),  # 6+4+3
    (7, 4, 2, 1.0, 0.6, 0.3, 0.0, 0.0, 0.0),    # 7+4+2
    (5, 5, 3, 1.0, 0.6, 0.3, 0.0, 0.0, 0.0),    # 5+5+3
    (4, 5, 4, 1.0, 0.6, 0.3, 0.0, 0.0, 0.0),    # 4+5+4
    (10, 3, 0, 1.0, 0.4, 0.0, 0.0, 0.0, 0.0),   # 10+3
    (11, 2, 0, 1.0, 0.3, 0.0, 0.0, 0.0, 0.0),   # 11+2
    (12, 1, 0, 1.0, 0.2, 0.0, 0.0, 0.0, 0.0),   # 12+1
]


def _triangle_areas(pts):
    """Vectorized computation of all triangle areas."""
    pi = pts[_triple_i]
    pj = pts[_triple_j]
    pk = pts[_triple_k]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    return 0.5 * np.abs(cross)


def _build_symmetric_points(n1, n2, n3, r1, r2, r3, phase1, phase2, phase3):
    """Build 13 points from concentric regular polygons with given parameters.
    
    Args:
        n1, n2, n3: number of points in each ring (sum = 13)
        r1, r2, r3: radii of each ring
        phase1, phase2, phase3: rotation angles for each ring
    
    Returns:
        array of shape (13, 2)
    """
    pts_list = []
    
    if n1 > 0:
        angles1 = np.linspace(0, 2 * np.pi, n1, endpoint=False) + phase1
        pts1 = np.column_stack([
            r1 * np.cos(angles1),
            r1 * np.sin(angles1)
        ])
        pts_list.append(pts1)
    
    if n2 > 0:
        angles2 = np.linspace(0, 2 * np.pi, n2, endpoint=False) + phase2
        pts2 = np.column_stack([
            r2 * np.cos(angles2),
            r2 * np.sin(angles2)
        ])
        pts_list.append(pts2)
    
    if n3 > 0:
        angles3 = np.linspace(0, 2 * np.pi, n3, endpoint=False) + phase3
        pts3 = np.column_stack([
            r3 * np.cos(angles3),
            r3 * np.sin(angles3)
        ])
        pts_list.append(pts3)
    
    pts = np.vstack(pts_list)
    return pts[:N]


def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0


def _score_points(pts):
    """Compute min triangle area / convex hull area for given points."""
    n = len(pts)
    if n < 3:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 0.0
    
    areas = _triangle_areas(pts)
    min_area = np.min(areas)
    
    if min_area < 1e-15:
        return 0.0
    
    return min_area / hull_area


def _logsumexp_surrogate(pts, temperature=0.01):
    """
    Smooth surrogate: -logsumexp(-areas/temperature) / hull_area
    This is differentiable and approximates min(areas)/hull_area.
    As temperature -> 0, it converges to the exact objective.
    """
    pts = np.asarray(pts, dtype=np.float64)
    hull_area = _hull_area(pts)
    if hull_area <= 1e-12:
        return 1e10
    
    areas = _triangle_areas(pts)
    # Use log-sum-exp for numerical stability
    scaled = areas / temperature
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    # -logsumexp(-areas/T) = -(max(-areas/T) + log(sum(exp(-areas/T - max))))
    # Actually: -logsumexp(-x) where x = areas/T
    # = -(max(-x) + log(sum(exp(-x - max(-x)))))
    # = -(-max(x) + log(sum(exp(x - max(x)))))  ... wait let me redo
    
    # We want a smooth approximation of min(areas)
    # min(x) ≈ -1/T * log(sum(exp(-x/T)))
    # So: -min(areas)/hull_area ≈ (1/T) * log(sum(exp(-areas/T))) / hull_area
    # For minimization, we minimize: log(sum(exp(-areas/T))) / (T * hull_area)
    
    neg_scaled = -areas / temperature
    max_neg = np.max(neg_scaled)
    log_sum_exp = max_neg + np.log(np.sum(np.exp(neg_scaled - max_neg)))
    
    return log_sum_exp / (temperature * hull_area)


def _neg_score(params):
    """Negative score for minimization. params is 1D array of length 2*N."""
    pts = params.reshape(N, 2)
    return -_score_points(pts)


def _neg_surrogate(params, temperature=0.01):
    """Negative surrogate for minimization."""
    pts = params.reshape(N, 2)
    return _logsumexp_surrogate(pts, temperature)


def _regular_polygon(n_points, radius=1.0, center=(0.0, 0.0)):
    """Generate vertices of a regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    pts = np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles)
    ])
    return pts


def _make_two_layer(n_outer=9, n_inner=4, r_outer=1.0, r_inner=0.5, angle_offset=0.0):
    """Two concentric rings: outer regular polygon + inner regular polygon."""
    return _build_symmetric_points(n_outer, n_inner, 0, r_outer, r_inner, 0.0, 0.0, angle_offset, 0.0)


def _make_three_layer(n1=5, n2=4, n3=4, r1=1.0, r2=0.6, r3=0.3):
    """Three concentric rings."""
    return _build_symmetric_points(n1, n2, n3, r1, r2, r3, 0.0, 0.0, 0.0)


def _normalize_to_unit_area(pts):
    """Scale points so convex hull has unit area."""
    hull_area = _hull_area(pts)
    if hull_area > 1e-12:
        scale = np.sqrt(1.0 / hull_area)
        return pts * scale
    return pts


def _optimize_from_init(init_pts, t_start, method='surrogate_then_exact'):
    """
    Multi-phase optimization:
    1. Optimize the smooth surrogate with Powell/L-BFGS-B
    2. Refine with exact objective
    """
    t0 = t_start
    best_params = init_pts.flatten().copy()
    best_score = _score_points(init_pts)
    
    # Phase 1: Surrogate optimization (smooth, differentiable)
    if (time.time() - t0) < (DEADLINE - 30.0):
        for temp in [0.05, 0.02, 0.01, 0.005]:
            if (time.time() - t0) > (DEADLINE - 25.0):
                break
            try:
                res = minimize(
                    _neg_surrogate,
                    best_params,
                    args=(temp,),
                    method='Powell',
                    options={'maxiter': 200, 'maxfun': 5000, 'xtol': 1e-8, 'ftol': 1e-8}
                )
                candidate = res.x.reshape(N, 2)
                score = _score_points(candidate)
                if score > best_score:
                    best_score = score
                    best_params = res.x.copy()
            except Exception:
                pass
    
    # Phase 2: Exact objective refinement with L-BFGS-B (using finite differences)
    if (time.time() - t0) < (DEADLINE - 15.0):
        try:
            res2 = minimize(
                _neg_score,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 300, 'maxfun': 10000, 'ftol': 1e-12}
            )
            candidate = res2.x.reshape(N, 2)
            score = _score_points(candidate)
            if score > best_score:
                best_score = score
                best_params = res2.x.copy()
        except Exception:
            pass
    
    # Phase 3: Powell on exact objective (handles non-differentiability)
    if (time.time() - t0) < (DEADLINE - 10.0):
        try:
            res3 = minimize(
                _neg_score,
                best_params,
                method='Powell',
                options={'maxiter': 100, 'maxfun': 5000, 'xtol': 1e-10, 'ftol': 1e-10}
            )
            candidate = res3.x.reshape(N, 2)
            score = _score_points(candidate)
            if score > best_score:
                best_score = score
                best_params = res3.x.copy()
        except Exception:
            pass
    
    return best_params, best_score


def _optimize_symmetric_config(n1, n2, n3, r1_init, r2_init, r3_init, t_start):
    """Optimize a symmetry-constrained configuration using coarse-to-fine grid search.
    
    Parameters to optimize:
    - r1, r2, r3: radii (3 params)
    - phase1, phase2, phase3: rotation angles (3 params)
    Total: 6 parameters
    
    Returns: (best_params_26d, best_score)
    """
    t0 = t_start
    best_params = None
    best_score = 0.0
    
    # Define search ranges
    # Radii: r1 is outermost, r2 middle, r3 innermost
    # Scale so that r1 ~ 1.0, r2 < r1, r3 < r2
    r1_range = (0.8, 1.2)
    r2_range = (0.2, 0.9) if n2 > 0 else (0.0, 0.0)
    r3_range = (0.0, 0.7) if n3 > 0 else (0.0, 0.0)
    
    # Phases: rotation angles, range [0, 2*pi/n] for each ring
    p1_range = (0.0, 2 * np.pi / n1) if n1 > 0 else (0.0, 0.0)
    p2_range = (0.0, 2 * np.pi / n2) if n2 > 0 else (0.0, 0.0)
    p3_range = (0.0, 2 * np.pi / n3) if n3 > 0 else (0.0, 0.0)
    
    # Coarse-to-fine grid search
    # Start with coarse grid, then refine around best point
    for n_grid in [5, 10, 20]:
        if (time.time() - t0) > (DEADLINE - 20.0):
            break
        
        # Create grid points for each parameter
        r1_vals = np.linspace(r1_range[0], r1_range[1], n_grid)
        r2_vals = np.linspace(r2_range[0], r2_range[1], n_grid) if n2 > 0 else np.array([0.0])
        r3_vals = np.linspace(r3_range[0], r3_range[1], n_grid) if n3 > 0 else np.array([0.0])
        p1_vals = np.linspace(p1_range[0], p1_range[1], n_grid) if n1 > 0 else np.array([0.0])
        p2_vals = np.linspace(p2_range[0], p2_range[1], n_grid) if n2 > 0 else np.array([0.0])
        p3_vals = np.linspace(p3_range[0], p3_range[1], n_grid) if n3 > 0 else np.array([0.0])
        
        # Evaluate all combinations (vectorized where possible)
        # For efficiency, we'll do nested loops but break early if time is running out
        best_local_score = 0.0
        best_local_params = None
        
        for r1 in r1_vals:
            for r2 in r2_vals:
                for r3 in r3_vals:
                    for p1 in p1_vals:
                        for p2 in p2_vals:
                            for p3 in p3_vals:
                                if (time.time() - t0) > (DEADLINE - 15.0):
                                    break
                                
                                pts = _build_symmetric_points(n1, n2, n3, r1, r2, r3, p1, p2, p3)
                                score = _score_points(pts)
                                if score > best_local_score:
                                    best_local_score = score
                                    best_local_params = (r1, r2, r3, p1, p2, p3)
        
        if best_local_params is not None:
            if best_local_score > best_score:
                best_score = best_local_score
                # Update ranges to focus around best point for next iteration
                best_r1, best_r2, best_r3, best_p1, best_p2, best_p3 = best_local_params
                delta_r = 0.15 / n_grid
                delta_p = (2 * np.pi / max(n1, 1)) / n_grid
                
                r1_range = (max(0.3, best_r1 - 3 * delta_r), min(1.5, best_r1 + 3 * delta_r))
                if n2 > 0:
                    r2_range = (max(0.05, best_r2 - 3 * delta_r), min(best_r1 - 0.05, best_r2 + 3 * delta_r))
                if n3 > 0:
                    r3_range = (max(0.0, best_r3 - 3 * delta_r), min(best_r2 - 0.05 if n2 > 0 else best_r1 - 0.05, best_r3 + 3 * delta_r))
                if n1 > 0:
                    p1_range = (max(0.0, best_p1 - 3 * delta_p), min(2 * np.pi / n1, best_p1 + 3 * delta_p))
                if n2 > 0:
                    p2_range = (max(0.0, best_p2 - 3 * delta_p), min(2 * np.pi / n2, best_p2 + 3 * delta_p))
                if n3 > 0:
                    p3_range = (max(0.0, best_p3 - 3 * delta_p), min(2 * np.pi / n3, best_p3 + 3 * delta_p))
    
    # Final refinement using scipy minimize on the 6D parameter space
    if best_local_params is not None and (time.time() - t0) < (DEADLINE - 10.0):
        def neg_symmetric_score(params_6d):
            r1, r2, r3, p1, p2, p3 = params_6d
            pts = _build_symmetric_points(n1, n2, n3, r1, r2, r3, p1, p2, p3)
            return -_score_points(pts)
        
        initial_6d = np.array(best_local_params)
        bounds_6d = list(zip(r1_range, r2_range, r3_range, p1_range, p2_range, p3_range))
        
        try:
            res = minimize(
                neg_symmetric_score,
                initial_6d,
                bounds=bounds_6d,
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 2000, 'ftol': 1e-12}
            )
            if res.fun < -best_score:
                best_score = -res.fun
                best_params = _build_symmetric_points(n1, n2, n3, *res.x).flatten()
        except Exception:
            pass
    
    # If we found a good symmetric solution, use it
    if best_params is not None:
        return best_params, best_score
    
    # Fallback: use the best grid point
    if best_local_params is not None:
        pts = _build_symmetric_points(n1, n2, n3, *best_local_params)
        return pts.flatten(), best_score
    
    # No good solution found
    return None, 0.0


def construct():
    t_start = time.time()
    
    best_params = None
    best_score = 0.0
    
    # Phase 1: Symmetry-constrained optimization (reduced 6D search space)
    # This is the main structural change - exploit rotational symmetry
    for config in _SYMMETRY_CONFIGS:
        if (time.time() - t_start) > (DEADLINE - 30.0):
            break
        
        n1, n2, n3, r1_init, r2_init, r3_init, p1_init, p2_init, p3_init = config
        
        try:
            params, score = _optimize_symmetric_config(n1, n2, n3, r1_init, r2_init, r3_init, t_start)
            if params is not None and score > best_score:
                best_score = score
                best_params = params
        except Exception:
            pass
    
    # Phase 2: If symmetry-constrained search didn't find good solutions, try free-space optimization
    if (time.time() - t_start) < (DEADLINE - 25.0) and (best_score < 50.0 or best_params is None):
        # Generate diverse initializations for free-space search
        inits = []
        
        # Regular 13-gon
        poly13 = _regular_polygon(N, radius=1.0)
        inits.append(_normalize_to_unit_area(poly13))
        
        # Two-layer configurations
        for n_outer, n_inner, r_inner in [(9, 4, 0.5), (8, 5, 0.45), (7, 6, 0.55)]:
            two_layer = _make_two_layer(n_outer=n_outer, n_inner=n_inner, r_outer=1.0, r_inner=r_inner)
            inits.append(_normalize_to_unit_area(two_layer))
        
        # Three-layer configurations
        for n1, n2, n3, r2, r3 in [(5, 4, 4, 0.6, 0.3), (6, 4, 3, 0.55, 0.25)]:
            three_layer = _make_three_layer(n1=n1, n2=n2, n3=n3, r1=1.0, r2=r2, r3=r3)
            inits.append(_normalize_to_unit_area(three_layer))
        
        # Random initialization
        rng = np.random.RandomState(123)
        rand_pts = rng.uniform(-1, 1, size=(N, 2))
        inits.append(_normalize_to_unit_area(rand_pts))
        
        # Optimize from each initialization
        for idx, init in enumerate(inits):
            if (time.time() - t_start) > (DEADLINE - 20.0):
                break
            try:
                params, score = _optimize_from_init(init, t_start)
                if score > best_score:
                    best_score = score
                    best_params = params
            except Exception:
                pass
    
    # Phase 3: Final refinement if time permits
    if best_params is not None and (time.time() - t_start) < (DEADLINE - 8.0):
        try:
            res_final = minimize(
                _neg_score,
                best_params,
                method='Powell',
                options={'maxiter': 50, 'maxfun': 2000, 'xtol': 1e-10, 'ftol': 1e-10}
            )
            score_final = _score_points(res_final.x.reshape(N, 2))
            if score_final > best_score:
                best_score = score_final
                best_params = res_final.x.copy()
        except Exception:
            pass
    
    # Safety: if somehow we have no valid result, return regular polygon
    if best_params is None:
        poly13 = _regular_polygon(N, radius=1.0)
        best_params = _normalize_to_unit_area(poly13).flatten()
    
    return best_params.reshape(N, 2)
