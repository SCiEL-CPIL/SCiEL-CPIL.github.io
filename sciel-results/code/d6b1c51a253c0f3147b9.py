import numpy as np
import time
import sys
from scipy.optimize import differential_evolution, minimize

def _normalize(pts):
    centroid = pts.mean(axis=0)
    p = pts - centroid
    d = np.linalg.norm(p, axis=1)
    mean_d = d.mean()
    if mean_d <= 0.01:
        return None, None, None
    p = p / mean_d
    return p, d, mean_d

def _check_strictly_convex(p):
    n = len(p)
    if n < 3:
        return False
    for i in range(n):
        if np.linalg.norm(p[i] - p[(i + 1) % n]) < 1e-9:
            return False
    signs = []
    for i in range(n):
        p0 = p[i]
        p1 = p[(i + 1) % n]
        p2 = p[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < 1e-9:
            return False
        signs.append(1 if cross > 0 else -1)
    if signs.count(signs[0]) != n:
        return False
    angles = []
    for i in range(n):
        p0 = p[i]
        p1 = p[(i + 1) % n]
        p2 = p[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_a = np.clip(cos_a, -1, 1)
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        sin_a = cross / (np.linalg.norm(v1) * np.linalg.norm(v2))
        ang = np.arctan2(sin_a, cos_a)
        angles.append(ang)
    total_turn = sum(angles)
    expected = 2 * np.pi if signs[0] > 0 else -2 * np.pi
    if abs(abs(total_turn) - 2 * np.pi) > 1e-4:
        return False
    return True

def _vertex_score(p, v_idx):
    n = len(p)
    v = p[v_idx]
    d = np.array([np.linalg.norm(v - p[j]) for j in range(n) if j != v_idx])
    d_sorted = np.sort(d)
    if np.any(d_sorted < 0.05):
        return 0.0
    gaps = d_sorted[1:] - d_sorted[:-1]
    k = int(np.argmin(gaps))
    D = (d_sorted[k] + d_sorted[k + 1]) / 2
    if D < 0.1:
        return 0.0
    if k == 0:
        selected = d_sorted[0:4]
    elif k == 17:
        selected = d_sorted[15:19]
    else:
        selected = d_sorted[k - 1:k + 3]
    if len(selected) < 4:
        return 0.0
    raw = np.mean([max(di / D, D / di) for di in selected])
    if raw < 1e-12:
        return 0.0
    score = min(1.0, 1.0 / raw)
    return score

def _score_polygon(p):
    pn, _, _ = _normalize(p)
    if pn is None:
        return 0.0
    if not _check_strictly_convex(pn):
        return 0.0
    min_vs = 1.0
    for i in range(len(pn)):
        vs = _vertex_score(pn, i)
        if vs < min_vs:
            min_vs = vs
        if min_vs < 0.01:
            break
    return min_vs

def _smooth_surrogate(p):
    """
    Smooth surrogate for optimization.
    Penalizes non-convexity and minimizes the distance gap metric.
    """
    n = len(p)
    if n < 3:
        return 1e6
    
    centroid = p.mean(axis=0)
    p = p - centroid
    d = np.linalg.norm(p, axis=1)
    mean_d = d.mean()
    if mean_d < 0.05:
        return 1e6
    p = p / mean_d

    # Convexity check (smooth penalty)
    min_cross = np.inf
    for i in range(n):
        p0 = p[i]
        p1 = p[(i + 1) % n]
        p2 = p[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        if abs(cross) < min_cross:
            min_cross = abs(cross)
    
    if min_cross < 1e-3:
        return 1e6 + (1e-3 - min_cross) * 10000

    # Check if all crosses have same sign
    crosses = []
    for i in range(n):
        p0 = p[i]
        p1 = p[(i + 1) % n]
        p2 = p[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        crosses.append(v1[0] * v2[1] - v1[1] * v2[0])
    
    pos = sum(1 for c in crosses if c > 0)
    neg = sum(1 for c in crosses if c < 0)
    if pos != n and neg != n:
        return 1e6
    
    # Compute smooth distance gap metric
    total_penalty = 0.0
    for i in range(n):
        v = p[i]
        dists = np.array([np.linalg.norm(v - p[j]) for j in range(n) if j != i])
        d_sorted = np.sort(dists)
        
        # Penalty for distances too close to 0
        if d_sorted[0] < 0.05:
            total_penalty += (0.05 - d_sorted[0]) * 100
            continue
            
        gaps = d_sorted[1:] - d_sorted[:-1]
        k = int(np.argmin(gaps))
        D = (d_sorted[k] + d_sorted[k + 1]) / 2
        
        if D < 0.1:
            total_penalty += (0.1 - D) * 100
            continue
            
        if k == 0:
            selected = d_sorted[0:4]
        elif k == 17:
            selected = d_sorted[15:19]
        else:
            selected = d_sorted[k - 1:k + 3]
            
        if len(selected) < 4:
            total_penalty += 1.0
            continue
            
        raw = np.mean([max(di / D, D / di) for di in selected])
        if raw < 1e-12:
            total_penalty += 1.0
        else:
            total_penalty += max(0.0, 1.0 - 1.0 / raw)
            
    return total_penalty

def _project_to_convex(p):
    """
    Attempt to project a set of points to a strictly convex polygon
    by sorting them by angle around their centroid.
    """
    centroid = p.mean(axis=0)
    p = p - centroid
    angles = np.arctan2(p[:, 1], p[:, 0])
    idx = np.argsort(angles)
    p_sorted = p[idx]
    return p_sorted

def equidistant_points_polygon20() -> np.ndarray:
    start_time = time.time()
    deadline = start_time + 580.0
    
    n = 20
    
    # 1. Global Search with Differential Evolution
    best_p = None
    best_score = 0.0
    
    # Initial bounds: square [-1.5, 1.5]
    bounds = [(-1.5, 1.5)] * (2 * n)
    
    def objective(x):
        p = x.reshape(n, 2)
        # Project to convex to ensure we're evaluating a valid polygon shape
        # Note: DE explores the space, but we want to guide it towards convex shapes.
        # Sorting by angle is a strong projection.
        p_proj = _project_to_convex(p)
        return _smooth_surrogate(p_proj)
        
    try:
        res = differential_evolution(
            objective, 
            bounds, 
            maxiter=100, 
            popsize=15, 
            tol=0.01, 
            seed=42,
            polish=True,
            init='sobol'
        )
        if res.success or res.fun < 1e5:
            p_candidate = res.x.reshape(n, 2)
            p_candidate = _project_to_convex(p_candidate)
            s = _score_polygon(p_candidate)
            if s > best_score:
                best_score = s
                best_p = p_candidate.copy()
    except Exception:
        pass

    if time.time() > deadline:
        if best_p is not None:
            return best_p
        # Fallback to regular
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])

    # 2. Local Polish with Nelder-Mead
    if best_p is not None:
        def local_obj(x):
            p = x.reshape(n, 2)
            p_proj = _project_to_convex(p)
            return _smooth_surrogate(p_proj)
            
        try:
            res_local = minimize(
                local_obj, 
                best_p.flatten(), 
                method='Nelder-Mead', 
                options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            p_candidate = res_local.x.reshape(n, 2)
            p_candidate = _project_to_convex(p_candidate)
            s = _score_polygon(p_candidate)
            if s > best_score:
                best_score = s
                best_p = p_candidate.copy()
        except Exception:
            pass

    # 3. Multi-start Nelder-Mead from diverse seeds
    if time.time() < deadline:
        seeds = []
        # Regular
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        seeds.append(np.column_stack([np.cos(angles), np.sin(angles)]))
        
        # Perturbed regular
        rng = np.random.RandomState(123)
        for _ in range(5):
            p = seeds[0] + 0.2 * rng.randn(n, 2)
            seeds.append(p)
            
        # Star-like variations
        for amp in [0.1, 0.2, 0.3]:
            r = 1.0 + amp * np.sin(2 * angles)
            seeds.append(np.column_stack([r * np.cos(angles), r * np.sin(angles)]))
            r = 1.0 + amp * np.cos(3 * angles)
            seeds.append(np.column_stack([r * np.cos(angles), r * np.sin(angles)]))

        for seed in seeds:
            if time.time() > deadline:
                break
            seed_proj = _project_to_convex(seed)
            s = _score_polygon(seed_proj)
            if s > best_score:
                best_score = s
                best_p = seed_proj.copy()
            
            # Local polish
            def local_obj(x):
                p = x.reshape(n, 2)
                p_proj = _project_to_convex(p)
                return _smooth_surrogate(p_proj)
                
            try:
                res_local = minimize(
                    local_obj, 
                    seed_proj.flatten(), 
                    method='Nelder-Mead', 
                    options={'maxiter': 500, 'xatol': 1e-5, 'fatol': 1e-5}
                )
                p_candidate = res_local.x.reshape(n, 2)
                p_candidate = _project_to_convex(p_candidate)
                s = _score_polygon(p_candidate)
                if s > best_score:
                    best_score = s
                    best_p = p_candidate.copy()
            except Exception:
                pass

    if best_p is None:
        angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
        return np.column_stack([np.cos(angles), np.sin(angles)])
        
    return best_p

# EVOLVE-BLOCK-END
