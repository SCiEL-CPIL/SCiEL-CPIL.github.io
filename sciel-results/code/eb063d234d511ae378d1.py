# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import itertools

_DEADLINE = time.time() + 590.0

def _all_triangle_areas(points):
    """Compute all triangle areas vectorized."""
    n = len(points)
    if n < 3:
        return np.array([0.0])
    idx = np.array(list(itertools.combinations(range(n), 3)))
    a = points[idx[:, 0]]
    b = points[idx[:, 1]]
    c = points[idx[:, 2]]
    areas = 0.5 * np.abs(a[:, 0] * (b[:, 1] - c[:, 1])
                         + b[:, 0] * (c[:, 1] - a[:, 1])
                         + c[:, 0] * (a[:, 1] - b[:, 1]))
    return areas

def _min_triangle_area(points):
    areas = _all_triangle_areas(points)
    return float(np.min(areas)) if len(areas) > 0 else 0.0

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

def _epigraph_objective(params, beta=100.0):
    """Epigraph formulation with soft constraints.
    params = [x1,y1,...,x13,y13,t] where t is the min triangle area.
    Objective: -t/hull_area + penalty for triangles with area < t (soft).
    """
    n = 13
    pts = params[:2*n].reshape(n, 2)
    t = params[2*n]
    
    try:
        ha = _hull_area(pts)
        if ha <= 0:
            return 1e6
        areas = _all_triangle_areas(pts)
        if np.any(areas <= 0):
            return 1e6
        # Soft penalty: sum of max(0, t - area_i)^2 smoothed
        deficit = np.maximum(0, t - areas)
        soft_penalty = np.sum(deficit**2) * beta
        return -t / ha + soft_penalty
    except Exception:
        return 1e6

def _soft_min_objective(points_flat, beta=200.0):
    """Smooth surrogate: -soft_min(triangle_areas) / hull_area."""
    pts = points_flat.reshape(-1, 2)
    try:
        ha = _hull_area(pts)
        if ha <= 0:
            return 1e6
        areas = _all_triangle_areas(pts)
        if np.any(areas <= 0):
            return 1e6
        neg_beta_areas = -beta * areas
        max_val = np.max(neg_beta_areas)
        log_sum_exp = max_val + np.log(np.sum(np.exp(neg_beta_areas - max_val)))
        soft_min = -log_sum_exp / beta
        return -soft_min / ha
    except Exception:
        return 1e6

def _optimize_soft_min(points0, betas=(50, 100, 200, 400, 600), max_iter_per_beta=200):
    """Optimize using progressive beta annealing with L-BFGS-B."""
    pts = points0.astype(float).flatten()
    n = len(points0)
    bounds = [(0.05, 0.95)] * (2 * n)
    
    for beta in betas:
        if time.time() > _DEADLINE - 30:
            break
        try:
            result = minimize(
                _soft_min_objective,
                pts,
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter_per_beta, 'maxfun': max_iter_per_beta * 3}
            )
            if result.fun < _soft_min_objective(pts, beta):
                pts = result.x
        except Exception:
            pass
    
    return pts.reshape(-1, 2)

def _optimize_epigraph(points0, betas=(50, 100, 200, 400), max_iter_per_beta=150):
    """Optimize using epigraph formulation with L-BFGS-B."""
    n = len(points0)
    pts = points0.astype(float).flatten()
    # Initial t estimate
    t0 = _min_triangle_area(points0) * 0.9
    params = np.concatenate([pts, [t0]])
    
    bounds = [(0.05, 0.95)] * (2 * n) + [(0.001, 0.5)]
    
    for beta in betas:
        if time.time() > _DEADLINE - 30:
            break
        try:
            result = minimize(
                _epigraph_objective,
                params,
                args=(beta,),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter_per_beta, 'maxfun': max_iter_per_beta * 3}
            )
            if result.fun < _epigraph_objective(params, beta):
                params = result.x
        except Exception:
            pass
    
    return params[:2*n].reshape(n, 2)

def _polish(points0, max_iter=500):
    """Final Powell polish with tight tolerances."""
    pts = points0.astype(float).flatten()
    n = len(points0)
    bounds = [(0.05, 0.95)] * (2 * n)
    try:
        result = minimize(
            _negative_ratio,
            pts,
            method='Powell',
            bounds=bounds,
            options={'maxiter': max_iter, 'xtol': 1e-15, 'ftol': 1e-17, 'maxfev': max_iter * 15}
        )
        if result.fun < _negative_ratio(pts):
            return result.x.reshape(-1, 2)
    except Exception:
        pass
    return points0

def _polish_epigraph(points0, max_iter=300):
    """Polish using epigraph formulation."""
    n = len(points0)
    pts = points0.astype(float).flatten()
    t0 = _min_triangle_area(points0) * 0.95
    params = np.concatenate([pts, [t0]])
    
    bounds = [(0.05, 0.95)] * (2 * n) + [(0.001, 0.5)]
    
    try:
        result = minimize(
            _epigraph_objective,
            params,
            args=(400.0,),
            method='Powell',
            bounds=bounds,
            options={'maxiter': max_iter, 'xtol': 1e-14, 'ftol': 1e-16, 'maxfev': max_iter * 10}
        )
        if result.fun < _epigraph_objective(params, 400.0):
            return result.x[:2*n].reshape(n, 2)
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

def _regular_polygon(n, radius=0.5, center=(0.5, 0.5), angle_offset=0.0):
    angles = np.linspace(angle_offset, 2 * np.pi + angle_offset, n, endpoint=False)
    return np.column_stack([center[0] + radius * np.cos(angles),
                            center[1] + radius * np.sin(angles)])

def _concentric_rings(outer_n, inner_n, outer_r, inner_r, center=(0.5, 0.5), angle_offset=0.0):
    """Create points on concentric rings with angle offsets."""
    outer = _regular_polygon(outer_n, radius=outer_r, center=center, angle_offset=angle_offset)
    if inner_n > 0:
        # Offset inner ring by half the angular step
        inner_angle_offset = angle_offset + np.pi / inner_n
        inner = _regular_polygon(inner_n, radius=inner_r, center=center, angle_offset=inner_angle_offset)
        return np.vstack([outer, inner])
    return outer

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
    
    # Multiple structured starting configurations with angle offsets
    starts = []
    
    # 1. Regular 13-gon with various angle offsets
    for off in [0.0, np.pi/13, 2*np.pi/13]:
        starts.append(_regular_polygon(13, radius=0.48, center=(0.5, 0.5), angle_offset=off))
    
    # 2. Concentric rings: 9 outer + 4 inner
    for off in [0.0, np.pi/9]:
        starts.append(_concentric_rings(9, 4, 0.48, 0.22, center=(0.5, 0.5), angle_offset=off))
    
    # 3. Concentric rings: 10 outer + 3 inner
    for off in [0.0, np.pi/10]:
        starts.append(_concentric_rings(10, 3, 0.48, 0.20, center=(0.5, 0.5), angle_offset=off))
    
    # 4. Concentric rings: 11 outer + 2 inner
    for off in [0.0, np.pi/11]:
        starts.append(_concentric_rings(11, 2, 0.48, 0.18, center=(0.5, 0.5), angle_offset=off))
    
    # 5. Concentric rings: 12 outer + 1 inner
    starts.append(_concentric_rings(12, 1, 0.48, 0.15, center=(0.5, 0.5), angle_offset=0.0))
    
    # 6. Two rings: 7 outer + 6 inner
    starts.append(_concentric_rings(7, 6, 0.48, 0.28, center=(0.5, 0.5), angle_offset=0.0))
    
    # 7. Three rings: 5 outer + 4 middle + 4 inner
    outer5 = _regular_polygon(5, radius=0.48, center=(0.5, 0.5))
    mid4 = _regular_polygon(4, radius=0.32, center=(0.5, 0.5), angle_offset=np.pi/8)
    inner4 = _regular_polygon(4, radius=0.16, center=(0.5, 0.5))
    starts.append(np.vstack([outer5, mid4, inner4]))
    
    # 8. Random starts
    for _ in range(4):
        starts.append(rng.uniform(0.1, 0.9, (13, 2)))
    
    # For each start, try multiple optimization strategies
    for start_idx, start in enumerate(starts):
        if time.time() > _DEADLINE - 40:
            break
        
        # Strategy A: Soft-min with progressive beta annealing
        if time.time() < _DEADLINE - 80:
            sm_pts = _optimize_soft_min(start, betas=(50, 100, 200, 400, 600), max_iter_per_beta=120)
            _update(sm_pts)
        
        # Strategy B: Epigraph optimization
        if time.time() < _DEADLINE - 60:
            ep_pts = _optimize_epigraph(start, betas=(50, 100, 200, 400), max_iter_per_beta=100)
            _update(ep_pts)
        
        # Strategy C: Powell polish on best of A and B
        if time.time() < _DEADLINE - 50:
            # Pick the better of soft-min and epigraph results
            s_sm = _score(sm_pts)
            s_ep = _score(ep_pts)
            if s_sm >= s_ep:
                base_pts = sm_pts
            else:
                base_pts = ep_pts
            polish_pts = _polish(base_pts, max_iter=400)
            _update(polish_pts)
        
        # Strategy D: Epigraph polish
        if time.time() < _DEADLINE - 40:
            ep_polish_pts = _polish_epigraph(base_pts, max_iter=200)
            _update(ep_polish_pts)
        
        # Strategy E: Coordinate descent from original start
        if time.time() < _DEADLINE - 35:
            cd_pts, cd_score = _coordinate_descent(start, max_sweeps=60)
            _update(cd_pts)
        
        # Strategy F: Soft-min on coordinate descent result
        if time.time() < _DEADLINE - 45:
            sm_pts2 = _optimize_soft_min(cd_pts, betas=(100, 200, 400), max_iter_per_beta=80)
            _update(sm_pts2)
        
        # Strategy G: Final polish on best found so far
        if time.time() < _DEADLINE - 30:
            if best_points is not None:
                final_polish = _polish(best_points, max_iter=300)
                _update(final_polish)
    
    # Final global polish if time permits
    if best_points is not None and time.time() < _DEADLINE - 20:
        final = _polish(best_points, max_iter=500)
        _update(final)
    
    if best_points is None:
        best_points = _regular_polygon(13, radius=0.48, center=(0.5, 0.5))
    
    if not np.all(np.isfinite(best_points)):
        best_points = _regular_polygon(13, radius=0.48, center=(0.5, 0.5))
    
    return best_points

# EVOLVE-BLOCK-END
