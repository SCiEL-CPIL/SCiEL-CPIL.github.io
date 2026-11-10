# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

_DEADLINE = time.time() + 620.0

def _min_triangle_area(points):
    n = len(points)
    if n < 3:
        return 0.0
    idx = np.array(list(itertools.combinations(range(n), 3)))
    if len(idx) == 0:
        return 0.0
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                         + b[:, 0] * (c[:, 1] - a[:, 1])
                         + c[:, 0] * (a[:, 1] - b[:, 1]))
    return float(np.min(areas))

def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return float(hull.volume)
    except Exception:
        return 0.0

def _score(points):
    ha = _hull_area(points)
    if ha <= 0:
        return 0.0
    ma = _min_triangle_area(points)
    if ma <= 0:
        return 0.0
    return ma / ha

def _negative_ratio(points_flat):
    """Objective to minimize: -min_area/hull_area"""
    pts = points_flat.reshape(-1, 2)
    try:
        ha = _hull_area(pts)
        if ha <= 0:
            return 1e6
        ma = _min_triangle_area(pts)
        if ma <= 0:
            return 1e6
        return -ma / ha
    except Exception:
        return 1e6

def _soft_min_triangle_area(points, beta=200.0):
    """Soft-min of triangle areas using log-sum-exp for smoothness."""
    n = len(points)
    if n < 3:
        return 0.0
    idx = np.array(list(itertools.combinations(range(n), 3)))
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                         + b[:, 0] * (c[:, 1] - a[:, 1])
                         + c[:, 0] * (a[:, 1] - b[:, 1]))
    m = np.min(areas)
    if m < 1e-15:
        return m
    log_sum = np.log(np.sum(np.exp(-beta * (areas - m))))
    soft_min = m - np.log(np.sum(np.exp(-beta * (areas - m)))) / beta
    return float(soft_min)

def _soft_negative_ratio(points_flat, beta=200.0):
    """Smooth surrogate objective for L-BFGS-B."""
    pts = points_flat.reshape(-1, 2)
    try:
        ha = _hull_area(pts)
        if ha <= 0:
            return 1e6
        ma = _soft_min_triangle_area(pts, beta)
        if ma <= 0:
            return 1e6
        return -ma / ha
    except Exception:
        return 1e6

def _optimize_with_scipy(points0, max_iter=500, use_soft=False, beta=200.0):
    """Use scipy's L-BFGS-B to optimize all 13 points simultaneously."""
    pts = points0.astype(float).flatten()
    n = len(points0)
    bounds = [(0.05, 0.95)] * (2 * n)
    
    if use_soft:
        obj_func = lambda x: _soft_negative_ratio(x, beta)
    else:
        obj_func = _negative_ratio
    
    try:
        result = minimize(
            obj_func,
            pts,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'maxfun': max_iter * 3}
        )
        result_pts = result.x.reshape(-1, 2)
        if _score(result_pts) > _score(points0):
            return result_pts
    except Exception:
        pass
    return points0

def _powell_refine(points0, max_iter=500):
    """Use Powell's method for final refinement on exact objective."""
    pts = points0.astype(float).flatten()
    n = len(points0)
    bounds = [(0.05, 0.95)] * (2 * n)
    
    try:
        result = minimize(
            _negative_ratio,
            pts,
            method='Powell',
            bounds=bounds,
            options={'maxiter': max_iter, 'maxfev': max_iter * 10}
        )
        result_pts = result.x.reshape(-1, 2)
        if _score(result_pts) > _score(points0):
            return result_pts
    except Exception:
        pass
    return points0

def _coordinate_descent(points0, max_sweeps=200, step_sizes=None):
    """Cyclic coordinate descent on each point's x and y."""
    if step_sizes is None:
        step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
    
    pts = points0.copy()
    best_pts = pts.copy()
    best_score = _score(pts)
    n = len(pts)
    
    for sweep in range(max_sweeps):
        if time.time() > _DEADLINE - 20:
            break
        improved = False
        for i in range(n):
            x0, y0 = pts[i]
            best_x, best_y = x0, y0
            best_local_score = best_score
            
            for step in step_sizes:
                for dx, dy in [(step,0),(-step,0),(0,step),(0,-step),
                               (step*0.707,step*0.707),(-step*0.707,step*0.707),
                               (step*0.707,-step*0.707),(-step*0.707,-step*0.707)]:
                    nx, ny = x0 + dx, y0 + dy
                    if nx < 0.05 or nx > 0.95 or ny < 0.05 or ny > 0.95:
                        continue
                    new_pts = pts.copy()
                    new_pts[i] = [nx, ny]
                    s = _score(new_pts)
                    if s > best_local_score + 1e-10:
                        best_local_score = s
                        best_x, best_y = nx, ny
                        improved = True
            pts[i] = [best_x, best_y]
        
        cur_score = _score(pts)
        if cur_score > best_score + 1e-10:
            best_score = cur_score
            best_pts = pts.copy()
        if not improved:
            break
    
    return best_pts, best_score

def _regular_polygon(n, radius=0.5, center=(0.5, 0.5)):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.column_stack([center[0] + radius * np.cos(angles),
                            center[1] + radius * np.sin(angles)])

def _concentric_rings(rings):
    """Create points on concentric rings. rings = list of (count, radius)."""
    all_pts = []
    for count, radius in rings:
        if count > 0:
            angles = np.linspace(0, 2 * np.pi, count, endpoint=False)
            pts = np.column_stack([0.5 + radius * np.cos(angles),
                                   0.5 + radius * np.sin(angles)])
            all_pts.append(pts)
    if not all_pts:
        return np.array([[0.5, 0.5]])
    return np.vstack(all_pts)

def heilbronn_convex13() -> np.ndarray:
    n = 13
    rng = np.random.default_rng(seed=42)
    
    best_points = None
    best_score = -np.inf
    
    def _update(pts):
        nonlocal best_points, best_score
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()
    
    starts = []
    
    # 1. Regular 13-gon
    starts.append(_regular_polygon(13, radius=0.48, center=(0.5, 0.5)))
    
    # 2. Ring + interior: 9 outer, 4 inner
    outer9 = _regular_polygon(9, radius=0.48, center=(0.5, 0.5))
    angles4 = np.linspace(0, 2*np.pi, 4, endpoint=False)
    inner4 = np.column_stack([0.5 + 0.2*np.cos(angles4), 0.5 + 0.2*np.sin(angles4)])
    starts.append(np.vstack([outer9, inner4]))
    
    # 3. Ring + interior: 10 outer, 3 inner
    outer10 = _regular_polygon(10, radius=0.48, center=(0.5, 0.5))
    angles3 = np.linspace(0, 2*np.pi, 3, endpoint=False)
    inner3 = np.column_stack([0.5 + 0.18*np.cos(angles3), 0.5 + 0.18*np.sin(angles3)])
    starts.append(np.vstack([outer10, inner3]))
    
    # 4. Ring + interior: 11 outer, 2 inner
    outer11 = _regular_polygon(11, radius=0.48, center=(0.5, 0.5))
    inner2 = np.array([[0.5 + 0.15, 0.5], [0.5 - 0.15, 0.5]])
    starts.append(np.vstack([outer11, inner2]))
    
    # 5. Ring + interior: 12 outer, 1 inner
    outer12 = _regular_polygon(12, radius=0.48, center=(0.5, 0.5))
    inner1 = np.array([[0.5, 0.5]])
    starts.append(np.vstack([outer12, inner1]))
    
    # 6. All 13 on hull (regular)
    starts.append(_regular_polygon(13, radius=0.5, center=(0.5, 0.5)))
    
    # 7. Concentric rings: 7 outer + 6 inner
    starts.append(_concentric_rings([(7, 0.48), (6, 0.25)]))
    
    # 8. Concentric rings: 8 outer + 5 inner
    starts.append(_concentric_rings([(8, 0.48), (5, 0.28)]))
    
    # 9. Concentric rings: 6 outer + 4 mid + 3 inner
    starts.append(_concentric_rings([(6, 0.48), (4, 0.32), (3, 0.15)]))
    
    # 10. Concentric rings: 9 outer + 4 mid
    starts.append(_concentric_rings([(9, 0.48), (4, 0.22)]))
    
    # 11. Concentric rings: 5 outer + 5 mid + 3 inner
    starts.append(_concentric_rings([(5, 0.48), (5, 0.30), (3, 0.12)]))
    
    # 12. Random starts
    for _ in range(3):
        starts.append(rng.uniform(0.1, 0.9, (13, 2)))
    
    for idx, start in enumerate(starts):
        if time.time() > _DEADLINE - 40:
            break
        
        # Strategy A: Soft L-BFGS-B (smooth surrogate)
        if time.time() < _DEADLINE - 80:
            opt_pts = _optimize_with_scipy(start, max_iter=300, use_soft=True, beta=200.0)
            _update(opt_pts)
        
        # Strategy B: Exact L-BFGS-B on soft result
        if time.time() < _DEADLINE - 60:
            opt_pts2 = _optimize_with_scipy(opt_pts, max_iter=200, use_soft=False)
            _update(opt_pts2)
        
        # Strategy C: Coordinate descent
        if time.time() < _DEADLINE - 40:
            cd_pts, cd_score = _coordinate_descent(opt_pts2, max_sweeps=80)
            _update(cd_pts)
        
        # Strategy D: Powell refinement (exact)
        if time.time() < _DEADLINE - 25:
            pow_pts = _powell_refine(cd_pts, max_iter=300)
            _update(pow_pts)
        
        # Strategy E: Fine coordinate descent
        if time.time() < _DEADLINE - 20:
            fine_pts, fine_score = _coordinate_descent(pow_pts, max_sweeps=100, 
                                                        step_sizes=[0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002])
            _update(fine_pts)
        
        # Strategy F: Final Powell on best so far
        if time.time() < _DEADLINE - 15:
            final_pts = _powell_refine(fine_pts, max_iter=200)
            _update(final_pts)
    
    if best_points is None:
        best_points = _regular_polygon(13, radius=0.48, center=(0.5, 0.5))
    
    if not np.all(np.isfinite(best_points)):
        best_points = _regular_polygon(13, radius=0.48, center=(0.5, 0.5))
    
    return best_points

# EVOLVE-BLOCK-END
