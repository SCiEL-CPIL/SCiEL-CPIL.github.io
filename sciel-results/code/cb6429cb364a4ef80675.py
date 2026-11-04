"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from itertools import combinations

_TRIPLETS = np.array(list(combinations(range(N), 3)))
_N_TRIPLES = _TRIPLETS.shape[0]

def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas vectorized."""
    i, j, k = _TRIPLETS.T
    a = pts[i]
    b = pts[j]
    c = pts[k]
    areas = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    return areas

def _convex_hull_area(pts):
    """Compute area of convex hull."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 0.0
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 0.0
    return np.min(areas) / hull_area

def _soft_min_objective(x, beta=50.0):
    """Negative soft-min of (min triangle area / hull area) for minimization."""
    pts = x.reshape(-1, 2)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 1e6
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 1e6
    scaled = -areas * beta
    max_scaled = np.max(scaled)
    soft_min = -max_scaled / beta - np.log(np.sum(np.exp(scaled - max_scaled))) / beta
    return -soft_min / hull_area

def _exact_objective(x):
    """Negative exact score for minimization (non-smooth)."""
    pts = x.reshape(-1, 2)
    return -_score(pts)

def _elliptic_params_to_points(params):
    """Convert elliptic parameters to 2D points.
    params: [a, b, theta_0, ...angles for N points, ...radial scales for N points]
    First N points are on an ellipse, remaining are interior.
    Actually, let's use: all 13 points parameterized by angle on ellipse + radial scale.
    params shape: (2N,) -> for each point i: angle_i, radius_scale_i
    Point i = radius_scale_i * (a*cos(angle_i), b*sin(angle_i))
    """
    # We'll use a simpler approach: parameterize by angles and a shared ellipse
    # params: [a, b, angle_0, ..., angle_12]
    a = params[0]
    b = params[1]
    angles = params[2:]
    pts = np.column_stack([a * np.cos(angles), b * np.sin(angles)])
    return pts

def _elliptic_objective(params):
    """Objective for elliptic parameterization."""
    a = params[0]
    b = params[1]
    angles = params[2:]
    pts = np.column_stack([a * np.cos(angles), b * np.sin(angles)])
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 1e6
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 1e6
    return -np.min(areas) / hull_area

def _free_objective_with_hull_constraint(x):
    """Objective that penalizes non-convex configurations."""
    pts = x.reshape(-1, 2)
    hull_area = _convex_hull_area(pts)
    if hull_area <= 0:
        return 1e6
    areas = _triangle_areas(pts)
    if len(areas) == 0:
        return 1e6
    # Penalize if points are too close together
    min_dist = np.inf
    for i in range(N):
        for j in range(i+1, N):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < min_dist:
                min_dist = d
    if min_dist < 0.01:
        return 1e6
    return -np.min(areas) / hull_area

def _nelder_mead_polish(pts, max_iter=500, tol=1e-10):
    """Polish using Nelder-Mead on the exact (non-smooth) objective."""
    x0 = pts.flatten()
    try:
        result = minimize(
            _exact_objective,
            x0,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol, 'adaptive': True}
        )
        return result.x.reshape(-1, 2)
    except Exception:
        return pts

def _multi_nm_polish(pts, max_iter_per_pass=300, n_passes=3, tol=1e-12):
    """Multiple Nelder-Mead passes."""
    current = pts.copy()
    for i in range(n_passes):
        if time.time() > _DEADLINE - 5:
            break
        polished = _nelder_mead_polish(current, max_iter=max_iter_per_pass, tol=tol)
        score_new = _score(polished)
        score_old = _score(current)
        if score_new > score_old:
            current = polished
    return current

def _differential_evolution_search(bounds, maxiter=100, popsize=15, seed=42):
    """Run differential evolution on the free 2D problem."""
    def obj(x):
        return _exact_objective(x)
    
    try:
        result = differential_evolution(
            obj,
            bounds,
            maxiter=maxiter,
            popsize=popsize,
            seed=seed,
            tol=1e-12,
            mutation=(0.5, 1.5),
            recombination=0.8,
            polish=True,
            updating='immediate',
            workers=1
        )
        return result.x.reshape(-1, 2), result.fun
    except Exception:
        return None, 1e6

def _local_polish(pts, max_iter=500):
    """Local polish with Nelder-Mead."""
    return _nelder_mead_polish(pts, max_iter=max_iter, tol=1e-12)

def construct():
    global _DEADLINE
    start_time = time.time()
    _DEADLINE = start_time + 148
    
    best_score = 0.0
    best_pts = None
    
    # Phase 1: Differential evolution for global search
    # Bounds: each coordinate in [-2, 2]
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    try:
        de_pts, de_score = _differential_evolution_search(bounds, maxiter=80, popsize=12, seed=42)
        if de_pts is not None:
            de_score_val = _score(de_pts)
            if de_score_val > best_score:
                best_score = de_score_val
                best_pts = de_pts.copy()
    except Exception:
        pass
    
    # Phase 2: Elliptic parameterization optimization
    if time.time() < _DEADLINE - 30:
        # Optimize elliptic params: [a, b, angle_0, ..., angle_12]
        # a, b in [0.3, 2.0], angles in [0, 2*pi]
        ell_bounds = [(0.3, 2.0), (0.3, 2.0)] + [(0, 2*np.pi)] * N
        
        # Multi-start elliptic optimization
        for ell_seed in range(5):
            if time.time() > _DEADLINE - 25:
                break
            try:
                rng = np.random.RandomState(ell_seed + 1000)
                x0 = np.concatenate([
                    [rng.uniform(0.5, 1.5), rng.uniform(0.5, 1.5)],
                    rng.uniform(0, 2*np.pi, N)
                ])
                
                result = minimize(
                    _elliptic_objective,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                ell_pts = _elliptic_params_to_points(result.x)
                ell_score = _score(ell_pts)
                if ell_score > best_score:
                    best_score = ell_score
                    best_pts = ell_pts.copy()
            except Exception:
                continue
    
    # Phase 3: If we have a good point, do local polishing
    if best_pts is not None and time.time() < _DEADLINE - 10:
        polished = _multi_nm_polish(best_pts, max_iter_per_pass=400, n_passes=3, tol=1e-12)
        polished_score = _score(polished)
        if polished_score > best_score:
            best_score = polished_score
            best_pts = polished.copy()
    
    # Phase 4: More differential evolution with different seeds if time permits
    if time.time() < _DEADLINE - 15:
        for de_seed in [123, 456, 789]:
            if time.time() > _DEADLINE - 12:
                break
            try:
                de_pts2, _ = _differential_evolution_search(bounds, maxiter=50, popsize=10, seed=de_seed)
                if de_pts2 is not None:
                    de_score2 = _score(de_pts2)
                    if de_score2 > best_score:
                        best_score = de_score2
                        best_pts = de_pts2.copy()
                        # Quick polish
                        if time.time() < _DEADLINE - 8:
                            polished2 = _nelder_mead_polish(best_pts, max_iter=300, tol=1e-12)
                            p2_score = _score(polished2)
                            if p2_score > best_score:
                                best_score = p2_score
                                best_pts = polished2.copy()
            except Exception:
                continue
    
    # Phase 5: Final high-precision polish
    if best_pts is not None and time.time() < _DEADLINE - 3:
        final_polish = _nelder_mead_polish(best_pts, max_iter=1000, tol=1e-14)
        final_score = _score(final_polish)
        if final_score > best_score:
            best_score = final_score
            best_pts = final_polish.copy()
    
    # Fallback if nothing worked
    if best_pts is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    return best_pts
# EVOLVE-BLOCK-END
