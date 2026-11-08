"""Heilbronn triangles: place 11 points to maximize min triangle area."""
import time
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 11
_H = np.sqrt(3) / 2.0
_SQRT3 = np.sqrt(3)

# Precompute all C(11,3) = 165 triangle index triples
_combos = list(combinations(range(N), 3))
_idx_i = np.array([c[0] for c in _combos])
_idx_j = np.array([c[1] for c in _combos])
_idx_k = np.array([c[2] for c in _combos])


def _min_triangle_area(pts):
    """Compute the minimum area of any triangle formed by 3 of the N points.
    pts has shape (N, 2)."""
    pi = pts[_idx_i]
    pj = pts[_idx_j]
    pk = pts[_idx_k]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _project_to_triangle(pts):
    """Project points to stay inside the equilateral triangle.
    Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2).
    Uses vectorized iterative projection.
    """
    pts = np.array(pts, dtype=float)
    for _ in range(20):
        # y >= 0
        pts[:, 1] = np.maximum(pts[:, 1], 0)
        # x >= 0
        pts[:, 0] = np.maximum(pts[:, 0], 0)
        
        # y <= sqrt(3) * x  =>  x >= y/sqrt(3)
        # If violated: y > sqrt(3) * x. 
        # Project onto line y = sqrt(3) x.
        # The line is y - sqrt(3)x = 0. Normal is (-sqrt(3), 1).
        # But simpler: if y > sqrt(3)x, we are above the left edge.
        # Wait, the triangle is defined by:
        # 1. y >= 0
        # 2. y <= sqrt(3) x  (Left edge from (0,0) to (0.5, H)? No, (0,0) to (1,0) is bottom.
        #    Vertices: (0,0), (1,0), (0.5, H).
        #    Edge 1 (Left): (0,0) to (0.5, H). Slope H/0.5 = sqrt(3). Eq: y = sqrt(3)x.
        #    Edge 2 (Right): (1,0) to (0.5, H). Slope H/-0.5 = -sqrt(3). Eq: y = -sqrt(3)(x-1) = sqrt(3)(1-x).
        #    Edge 3 (Bottom): y=0.
        
        # So constraints:
        # y >= 0
        # y <= sqrt(3) x
        # y <= sqrt(3) (1 - x)
        
        # Violation 1: y > sqrt(3) x. Point is to the left/above the left edge.
        # Project onto y = sqrt(3) x.
        # Vector from (x,y) to line. Line: -sqrt(3)x + y = 0.
        # Distance = (-sqrt(3)x + y) / 2.
        # Normal n = (-sqrt(3), 1) / 2.
        # Projection P = (x,y) - dist * n? No, dist is signed.
        # Let's just use the iterative clamping which is robust enough for 11 points.
        
        mask2 = pts[:, 1] > _SQRT3 * pts[:, 0]
        if np.any(mask2):
            # Move point down/left to satisfy y = sqrt(3) x
            # Simplest: adjust y to be sqrt(3) x. This might violate y>=0 or other bounds, loop handles it.
            # Or adjust x to be y/sqrt(3).
            # Let's adjust y, as x is usually more "free" in the middle.
            pts[mask2, 1] = _SQRT3 * pts[mask2, 0]
            
        mask3 = pts[:, 1] > _SQRT3 * (1.0 - pts[:, 0])
        if np.any(mask3):
            pts[mask3, 1] = _SQRT3 * (1.0 - pts[mask3, 0])
            
        # Re-clip y >= 0 and x in [0,1]
        pts[:, 1] = np.maximum(pts[:, 1], 0)
        pts[:, 0] = np.clip(pts[:, 0], 0, 1)
        
    return pts


def _soft_min(values, beta=100.0):
    """Smooth approximation of the minimum using log-sum-exp."""
    min_val = np.min(values)
    shifted = values - min_val
    # Avoid overflow in exp
    max_shifted = np.max(shifted)
    if max_shifted > 50:
        shifted = shifted - max_shifted
    sum_exp = np.sum(np.exp(-beta * shifted))
    return min_val - np.log(sum_exp) / beta


def _penalty(points):
    """Penalty for points outside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    d1 = np.maximum(0, -y)
    viol2 = np.maximum(0, y - _SQRT3 * x)
    viol3 = np.maximum(0, y - (_SQRT3 - _SQRT3 * x))
    return np.sum(d1**2) + np.sum(viol2**2) + np.sum(viol3**2)


def _smooth_objective(x, beta=50.0):
    """
    Smooth surrogate: negative soft-min of triangle areas + penalty.
    """
    pts = x.reshape(N, 2)
    pi = pts[_idx_i]
    pj = pts[_idx_j]
    pk = pts[_idx_k]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    sm = _soft_min(areas, beta)
    pen = _penalty(pts)
    return -(sm) + 10.0 * pen


def _true_objective(x):
    """Negative min triangle area (to be minimized)."""
    pts = x.reshape(N, 2)
    pts = _project_to_triangle(pts)
    return -_min_triangle_area(pts)


def _symmetric_initial_config():
    """Create a symmetric initial configuration for 11 points."""
    cx, cy = 0.5, _H / 3
    r = 0.25
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    pts = [[cx, cy]]
    for a in angles:
        px = cx + r * np.cos(a)
        py = cy + r * np.sin(a)
        pts.append([px, py])
    pts.append([0.15, 0.1])
    pts.append([0.85, 0.1])
    pts.append([0.5, 0.7])
    pts.append([0.35, 0.35])
    arr = np.array(pts[:N], dtype=float)
    return _project_to_triangle(arr).flatten()


def _random_initial_config(seed):
    """Generate a random initial configuration."""
    rng = np.random.RandomState(seed)
    pts = []
    for _ in range(N):
        u, v = rng.random(), rng.random()
        while u + v > 1.0:
            u, v = rng.random(), rng.random()
        x = u + 0.5 * v
        y = _H * v
        pts.append([x, y])
    arr = np.array(pts, dtype=float)
    return _project_to_triangle(arr).flatten()


def _row_initial_config(seed, variant=0):
    """Generate row-based initial configurations."""
    rng = np.random.RandomState(seed)
    pts = []
    if variant == 0:
        # 3-3-3-2 pattern
        for x in [0.1, 0.5, 0.9]:
            pts.append([x, 0.01])
        for x in [0.2, 0.5, 0.8]:
            pts.append([x, _H / 3])
        for x in [0.3, 0.5, 0.7]:
            pts.append([x, 2 * _H / 3])
        pts.append([0.4, 0.95 * _H])
        pts.append([0.6, 0.95 * _H])
    elif variant == 1:
        # 4-3-3-1 pattern
        for x in [0.08, 0.36, 0.64, 0.92]:
            pts.append([x, 0.01])
        for x in [0.2, 0.5, 0.8]:
            pts.append([x, _H / 3])
        for x in [0.35, 0.5, 0.65]:
            pts.append([x, 2 * _H / 3])
        pts.append([0.5, 0.95 * _H])
    else:
        # 3-4-3-1 pattern
        for x in [0.15, 0.5, 0.85]:
            pts.append([x, 0.01])
        for x in [0.1, 0.37, 0.63, 0.9]:
            pts.append([x, _H / 3])
        for x in [0.3, 0.5, 0.7]:
            pts.append([x, 2 * _H / 3])
        pts.append([0.5, 0.95 * _H])
        
    pts = np.array(pts[:N], dtype=float)
    pts = pts + rng.uniform(-0.02, 0.02, size=(N, 2))
    return _project_to_triangle(pts).flatten()


def _lbfgsb_optimize(x0, max_time=8.0):
    """
    Run L-BFGS-B on the smooth surrogate objective.
    Returns (best_x, best_val_true).
    """
    start = time.time()
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, _H))
    
    best_x = x0.copy()
    best_val = _true_objective(best_x)
    
    # Multi-stage beta for soft-min
    # Start loose, get tight
    for beta in [20.0, 50.0, 100.0, 200.0, 400.0]:
        if time.time() > start + max_time - 1.0:
            break
        
        obj = lambda x: _smooth_objective(x, beta)
        
        try:
            res = minimize(
                obj,
                best_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 150, 'ftol': 1e-10, 'gtol': 1e-8, 'disp': False}
            )
            if res.fun < obj(best_x):
                cand = res.x
                cand_val = _true_objective(cand)
                if cand_val < best_val:
                    best_x = cand
                    best_val = cand_val
        except Exception:
            pass
    
    return best_x, best_val


def _polish_candidate(x0, max_time=10.0):
    """
    High-precision polishing of a candidate using SLSQP and Nelder-Mead
    on the true non-smooth objective, plus an epigraph-style focus.
    """
    start = time.time()
    best_x = x0.copy()
    best_val = _true_objective(best_x)
    
    bounds = []
    for i in range(N):
        bounds.append((0, 1))
        bounds.append((0, _H))
        
    # 1. SLSQP polish (good for handling constraints and kinks with finite diffs)
    if time.time() < start + max_time - 4.0:
        try:
            res = minimize(
                _true_objective,
                best_x,
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'disp': False}
            )
            if res.success or res.fun < best_val:
                cand = _project_to_triangle(res.x.reshape(N, 2)).flatten()
                cand_val = _true_objective(cand)
                if cand_val < best_val:
                    best_x = cand
                    best_val = cand_val
        except Exception:
            pass

    # 2. Nelder-Mead polish (robust local search)
    if time.time() < start + max_time - 2.0:
        try:
            res = minimize(
                _true_objective,
                best_x,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-9, 'fatol': 1e-11}
            )
            if res.fun < best_val:
                cand = _project_to_triangle(res.x.reshape(N, 2)).flatten()
                cand_val = _true_objective(cand)
                if cand_val < best_val:
                    best_x = cand
                    best_val = cand_val
        except Exception:
            pass
            
    # 3. Epigraph-style focus: Identify critical triangles and re-optimize
    if time.time() < start + max_time - 1.0:
        pts = _project_to_triangle(best_x.reshape(N, 2))
        pi = pts[_idx_i]
        pj = pts[_idx_j]
        pk = pts[_idx_k]
        cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
                (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
        areas = 0.5 * np.abs(cross)
        
        min_area = np.min(areas)
        # Triangles that are close to the minimum are the "bottlenecks"
        critical_mask = areas < min_area * 1.05
        critical_indices = np.where(critical_mask)[0]
        
        if len(critical_indices) > 0:
            # Create an objective that heavily penalizes these specific triangles
            def _focused_obj(x):
                p = x.reshape(N, 2)
                p = _project_to_triangle(p)
                pi = p[_idx_i]
                pj = p[_idx_j]
                pk = p[_idx_k]
                cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
                        (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
                a = 0.5 * np.abs(cross)
                # We want to maximize the min area. 
                # Here we minimize -min area. 
                # But we want to specifically push up the critical ones.
                # If we just use min, it might ignore the others.
                # Let's use a weighted sum of the critical ones + min of all.
                # Actually, just minimizing -min(a) is the goal. 
                # The "focus" idea is to help the gradient method see the kinks better?
                # L-BFGS-B on the true objective might struggle.
                # Let's try SLSQP on the true objective again with tighter tol.
                return -np.min(a)
            
            try:
                res = minimize(
                    _focused_obj,
                    best_x,
                    method='SLSQP',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-14, 'disp': False}
                )
                if res.fun < best_val:
                    cand = _project_to_triangle(res.x.reshape(N, 2)).flatten()
                    cand_val = _true_objective(cand)
                    if cand_val < best_val:
                        best_x = cand
                        best_val = cand_val
            except Exception:
                pass

    return best_x, best_val


def construct():
    deadline = time.time() + 145.0
    
    candidates = []
    
    # Phase 1: Generate diverse starts and run L-BFGS-B on soft-min surrogate
    # This is the main exploration phase
    
    # Symmetric start
    x0 = _symmetric_initial_config()
    x, val = _lbfgsb_optimize(x0, max_time=10.0)
    candidates.append((x, val))
    
    # Row-based starts
    for variant in range(3):
        if time.time() > deadline - 60:
            break
        x0 = _row_initial_config(42 + variant, variant=variant)
        x, val = _lbfgsb_optimize(x0, max_time=8.0)
        candidates.append((x, val))
        
    # Random starts
    for seed in range(5):
        if time.time() > deadline - 40:
            break
        x0 = _random_initial_config(100 + seed)
        x, val = _lbfgsb_optimize(x0, max_time=5.0)
        candidates.append((x, val))
        
    # Sort candidates by value (lower is better)
    candidates.sort(key=lambda t: t[1])
    
    # Keep top 3 unique candidates (simple check: if distance is small, skip)
    top_candidates = []
    for x, val in candidates:
        is_dup = False
        for tx, tv in top_candidates:
            if np.linalg.norm(x - tx) < 1e-4:
                is_dup = True
                break
        if not is_dup:
            top_candidates.append((x, val))
        if len(top_candidates) == 3:
            break
            
    best_x = top_candidates[0][0] if top_candidates else x
    best_val = top_candidates[0][1] if top_candidates else val
    
    # Phase 2: Polish the top candidates
    # We have about 40-50 seconds left.
    # Spend ~15s on the best, ~10s on the second, ~5s on the third.
    
    if len(top_candidates) > 0:
        # Polish best
        t_start = time.time()
        max_time_1 = min(15.0, deadline - t_start - 20.0)
        if max_time_1 > 2.0:
            x, val = _polish_candidate(top_candidates[0][0], max_time=max_time_1)
            if val < best_val:
                best_x, best_val = x, val
                
    if len(top_candidates) > 1:
        t_start = time.time()
        max_time_2 = min(10.0, deadline - t_start - 10.0)
        if max_time_2 > 2.0:
            x, val = _polish_candidate(top_candidates[1][0], max_time=max_time_2)
            if val < best_val:
                best_x, best_val = x, val
                
    if len(top_candidates) > 2:
        t_start = time.time()
        max_time_3 = min(5.0, deadline - t_start - 5.0)
        if max_time_3 > 1.0:
            x, val = _polish_candidate(top_candidates[2][0], max_time=max_time_3)
            if val < best_val:
                best_x, best_val = x, val

    # Final micro-polish on the absolute best
    if time.time() < deadline - 2.0:
        t_start = time.time()
        max_time_final = min(2.0, deadline - t_start - 0.5)
        if max_time_final > 0.5:
             # Quick Nelder-Mead
             try:
                 res = minimize(
                     _true_objective,
                     best_x,
                     method='Nelder-Mead',
                     options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12}
                 )
                 if res.fun < best_val:
                     cand = _project_to_triangle(res.x.reshape(N, 2)).flatten()
                     cand_val = _true_objective(cand)
                     if cand_val < best_val:
                         best_x = cand
                         best_val = cand_val
             except Exception:
                 pass

    pts = _project_to_triangle(best_x.reshape(N, 2))
    return pts
