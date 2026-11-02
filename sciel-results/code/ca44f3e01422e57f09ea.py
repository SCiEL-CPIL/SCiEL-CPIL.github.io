"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 13

def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples, vectorized."""
    n = len(pts)
    if n < 3:
        return 0.0
    min_area = float('inf')
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            v1 = pts[j] - pts[i]
            v2 = pts[j+1:] - pts[i]
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            if len(cross) > 0:
                local_min = 0.5 * np.min(cross)
                if local_min < min_area:
                    min_area = local_min
    return min_area if min_area != float('inf') else 0.0

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _score(pts):
    """Score = min triangle area / convex hull area."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    return _min_triangle_area(pts) / hull_a

def _neg_score(params):
    pts = params.reshape(-1, 2)
    return -_score(pts)

def _soft_min_triangle_area(pts, alpha=50.0):
    """
    Smooth approximation of the minimum triangle area.
    Uses -log(sum(exp(-alpha * areas))) / alpha
    """
    n = len(pts)
    if n < 3:
        return 0.0
    all_areas = []
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            v1 = pts[j] - pts[i]
            v2 = pts[j+1:] - pts[i]
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            if len(cross) > 0:
                all_areas.append(0.5 * cross)
    if not all_areas:
        return 0.0
    areas = np.concatenate(all_areas)
    if len(areas) == 0:
        return 0.0
    min_a = np.min(areas)
    shifted = -alpha * (areas - min_a)
    max_shifted = np.max(shifted)
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted))) - alpha * min_a
    return -log_sum / alpha

def _smooth_score(pts, alpha=50.0):
    """Smooth score surrogate."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    smooth_min = _soft_min_triangle_area(pts, alpha)
    return smooth_min / hull_a

def _neg_smooth_score(params, alpha=50.0):
    pts = params.reshape(-1, 2)
    s = _smooth_score(pts, alpha)
    if not np.isfinite(s) or s < 0:
        return 1e10
    return -s

def _make_symmetric_points(params):
    """
    Construct 13 points from symmetric parameters.
    params: [r_outer, r_inner, theta_outer, theta_inner, center_x, center_y]
    Structure: 6 outer points (hexagon), 6 inner points (hexagon), 1 center point.
    """
    r_out, r_in, th_out, th_in, cx, cy = params
    angles_out = np.linspace(0, 2 * np.pi, 6, endpoint=False) + th_out
    angles_in = np.linspace(0, 2 * np.pi, 6, endpoint=False) + th_in
    
    pts_out = np.column_stack([r_out * np.cos(angles_out), r_out * np.sin(angles_out)])
    pts_in = np.column_stack([r_in * np.cos(angles_in), r_in * np.sin(angles_in)])
    pts_c = np.array([[cx, cy]])
    
    return np.vstack([pts_out, pts_in, pts_c])

def _neg_symmetric_score(params):
    pts = _make_symmetric_points(params)
    return -_score(pts)

def _neg_symmetric_smooth_score(params, alpha=50.0):
    pts = _make_symmetric_points(params)
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 1e10
    smooth_min = _soft_min_triangle_area(pts, alpha)
    val = smooth_min / hull_a
    if not np.isfinite(val) or val < 0:
        return 1e10
    return -val

def _make_concentric_points(params):
    """
    Construct 13 points from concentric ring parameters.
    params: [r1, r2, r3, th1, th2, th3]
    Structure: 5 points on ring 1, 5 points on ring 2, 3 points on ring 3.
    """
    r1, r2, r3, th1, th2, th3 = params
    a1 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + th1
    a2 = np.linspace(0, 2 * np.pi, 5, endpoint=False) + th2
    a3 = np.linspace(0, 2 * np.pi, 3, endpoint=False) + th3
    
    p1 = np.column_stack([r1 * np.cos(a1), r1 * np.sin(a1)])
    p2 = np.column_stack([r2 * np.cos(a2), r2 * np.sin(a2)])
    p3 = np.column_stack([r3 * np.cos(a3), r3 * np.sin(a3)])
    
    return np.vstack([p1, p2, p3])

def _neg_concentric_score(params):
    pts = _make_concentric_points(params)
    return -_score(pts)

def _neg_concentric_smooth_score(params, alpha=50.0):
    pts = _make_concentric_points(params)
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 1e10
    smooth_min = _soft_min_triangle_area(pts, alpha)
    val = smooth_min / hull_a
    if not np.isfinite(val) or val < 0:
        return 1e10
    return -val

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit
    
    best_score = 0.0
    best_pts = np.zeros((N, 2))
    rng = np.random.RandomState(42)
    
    # 1. Generate diverse initializations
    inits = []
    
    # Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]))
    
    # 12-gon + center
    m = N - 1
    angles_m = np.linspace(0, 2 * np.pi, m, endpoint=False)
    outer = np.column_stack([np.cos(angles_m), np.sin(angles_m)])
    center = np.array([[0.0, 0.0]])
    inits.append(np.vstack([outer, center]))
    
    # Concentric rings (6+6+1) - Symmetric hexagonal structure
    for r_in in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for th_in in [0.0, np.pi/6.0, np.pi/12.0]:
            params = [1.0, r_in, 0.0, th_in, 0.0, 0.0]
            pts = _make_symmetric_points(params)
            inits.append(pts)
    
    # Concentric rings (5+5+3) - Alternative structure
    for r2 in [0.5, 0.7]:
        for r3 in [0.3, 0.8]:
            params = [1.0, r2, r3, 0.0, np.pi/5.0, 0.0]
            pts = _make_concentric_points(params)
            inits.append(pts)
    
    # Jittered regular polygon
    for jitter in [0.01, 0.05, 0.1]:
        r = 1.0 + jitter * rng.randn(N)
        inits.append(np.column_stack([r * np.cos(angles), r * np.sin(angles)]))
        
    # Random points in unit disk
    for _ in range(5):
        r = rng.rand(N) ** 0.5
        theta = rng.rand(N) * 2 * np.pi
        inits.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
    
    # Additional structured initializations
    # 4+5+4 rings
    for r_mid in [0.5, 0.6, 0.7]:
        a1 = np.linspace(0, 2*np.pi, 4, endpoint=False)
        a2 = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
        a3 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
        p1 = np.column_stack([np.cos(a1), np.sin(a1)])
        p2 = np.column_stack([r_mid * np.cos(a2), r_mid * np.sin(a2)])
        p3 = np.column_stack([0.3 * np.cos(a3), 0.3 * np.sin(a3)])
        inits.append(np.vstack([p1, p2, p3]))
    
    # 7+6 split
    for ratio in [0.4, 0.5, 0.6]:
        a1 = np.linspace(0, 2*np.pi, 7, endpoint=False)
        a2 = np.linspace(0, 2*np.pi, 6, endpoint=False) + np.pi/6
        p1 = np.column_stack([np.cos(a1), np.sin(a1)])
        p2 = np.column_stack([ratio * np.cos(a2), ratio * np.sin(a2)])
        inits.append(np.vstack([p1, p2]))
    
    # Evaluate initializations
    scored_inits = []
    for init_pts in inits:
        if time.time() > deadline - 50.0:
            break
        s = _score(init_pts)
        scored_inits.append((s, init_pts))
        if s > best_score:
            best_score = s
            best_pts = init_pts.copy()
            
    scored_inits.sort(key=lambda x: -x[0])
    
    # 2. Optimization Loop with multiple alphas for smooth refinement
    candidates = []
    for s, p in scored_inits[:8]:
        candidates.append(p)
    candidates.append(best_pts)
    
    # Remove duplicates roughly
    unique_candidates = []
    for c in candidates:
        if not any(np.allclose(c, u, atol=1e-5) for u in unique_candidates):
            unique_candidates.append(c)
            
    for init_pts in unique_candidates:
        if time.time() > deadline - 30.0:
            break
            
        current_pts = init_pts.copy()
        current_score = _score(current_pts)
        
        # Step A: Multi-alpha Smooth Refinement
        for alpha in [30.0, 50.0, 100.0]:
            if time.time() > deadline - 25.0:
                break
            try:
                res_smooth = minimize(
                    lambda p, a=alpha: _neg_smooth_score(p, a),
                    current_pts.flatten(),
                    method='Powell',
                    options={'maxiter': 300, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                smooth_pts = res_smooth.x.reshape(-1, 2)
                smooth_score = _score(smooth_pts)
                if smooth_score > current_score:
                    current_pts = smooth_pts
                    current_score = smooth_score
            except Exception:
                pass
        
        # Step B: Exact Refinement with Nelder-Mead
        if time.time() > deadline - 15.0:
            continue
        try:
            res_exact = minimize(
                _neg_score,
                current_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            exact_pts = res_exact.x.reshape(-1, 2)
            exact_score = _score(exact_pts)
            if exact_score > current_score:
                current_pts = exact_pts
                current_score = exact_score
        except Exception:
            pass
        
        if current_score > best_score:
            best_score = current_score
            best_pts = current_pts.copy()
            
    # 3. Symmetric Optimization
    if time.time() < deadline - 35.0:
        # Hexagonal symmetric space (6+6+1)
        sym_inits = [
            [1.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.7, 0.0, np.pi/6.0, 0.0, 0.0],
            [1.0, 0.8, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.6, 0.0, np.pi/12.0, 0.0, 0.0],
            [1.0, 0.4, 0.0, np.pi/6.0, 0.0, 0.0],
            [1.0, 0.55, 0.0, np.pi/12.0, 0.0, 0.0],
            [1.0, 0.65, 0.0, np.pi/6.0, 0.0, 0.0],
            [1.0, 0.45, 0.0, np.pi/6.0, 0.0, 0.0],
            [1.0, 0.55, 0.0, 0.0, 0.0, 0.0],
        ]
        # Add jittered versions
        for _ in range(10):
            sym_inits.append([
                1.0 + 0.05*rng.randn(),
                0.5 + 0.15*rng.randn(),
                0.05*rng.randn(),
                0.05*rng.randn(),
                0.02*rng.randn(),
                0.02*rng.randn()
            ])
            
        for sym_init in sym_inits:
            if time.time() > deadline - 20.0:
                break
            
            current_sym_params = np.array(sym_init)
            
            # Smooth refinement in symmetric space with multiple alphas
            best_sym_score = -np.inf
            best_sym_params = current_sym_params.copy()
            for alpha in [30.0, 50.0, 80.0]:
                if time.time() > deadline - 18.0:
                    break
                try:
                    res_sym_smooth = minimize(
                        lambda p, a=alpha: _neg_symmetric_smooth_score(p, a),
                        current_sym_params,
                        method='L-BFGS-B',
                        options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    sym_params = res_sym_smooth.x
                    sym_pts = _make_symmetric_points(sym_params)
                    sym_score = _score(sym_pts)
                    if sym_score > best_sym_score:
                        best_sym_score = sym_score
                        best_sym_params = sym_params.copy()
                except Exception:
                    pass
            
            if best_sym_score > -np.inf and best_sym_score > best_score:
                best_score = best_sym_score
                best_pts = _make_symmetric_points(best_sym_params).copy()
            
            # Exact refinement in symmetric space
            if time.time() > deadline - 10.0:
                continue
            try:
                res_sym_exact = minimize(
                    _neg_symmetric_score,
                    best_sym_params,
                    method='Nelder-Mead',
                    options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                sym_params_exact = res_sym_exact.x
                sym_pts_exact = _make_symmetric_points(sym_params_exact)
                sym_score_exact = _score(sym_pts_exact)
                
                if sym_score_exact > best_score:
                    best_score = sym_score_exact
                    best_pts = sym_pts_exact.copy()
            except Exception:
                pass

        # Concentric 5+5+3 symmetric space
        if time.time() < deadline - 20.0:
            conc_inits = [
                [1.0, 0.6, 0.3, 0.0, np.pi/5.0, 0.0],
                [1.0, 0.7, 0.4, 0.0, 0.0, 0.0],
                [1.0, 0.5, 0.2, np.pi/10.0, np.pi/5.0, np.pi/6.0],
                [1.0, 0.65, 0.35, 0.0, np.pi/5.0, np.pi/3.0],
                [1.0, 0.55, 0.25, np.pi/10.0, 0.0, np.pi/6.0],
            ]
            for _ in range(6):
                conc_inits.append([
                    1.0 + 0.05*rng.randn(),
                    0.6 + 0.15*rng.randn(),
                    0.3 + 0.15*rng.randn(),
                    0.05*rng.randn(),
                    0.05*rng.randn(),
                    0.05*rng.randn()
                ])
                
            for conc_init in conc_inits:
                if time.time() > deadline - 12.0:
                    break
                current_conc_params = np.array(conc_init)
                best_conc_score = -np.inf
                best_conc_params = current_conc_params.copy()
                
                for alpha in [30.0, 50.0, 80.0]:
                    if time.time() > deadline - 10.0:
                        break
                    try:
                        res_conc_smooth = minimize(
                            lambda p, a=alpha: _neg_concentric_smooth_score(p, a),
                            current_conc_params,
                            method='L-BFGS-B',
                            options={'maxiter': 150, 'ftol': 1e-12, 'gtol': 1e-8}
                        )
                        conc_params = res_conc_smooth.x
                        conc_pts = _make_concentric_points(conc_params)
                        conc_score = _score(conc_pts)
                        if conc_score > best_conc_score:
                            best_conc_score = conc_score
                            best_conc_params = conc_params.copy()
                    except Exception:
                        pass
                
                if best_conc_score > -np.inf and best_conc_score > best_score:
                    best_score = best_conc_score
                    best_pts = _make_concentric_points(best_conc_params).copy()
                
                if time.time() > deadline - 5.0:
                    continue
                try:
                    res_conc_exact = minimize(
                        _neg_concentric_score,
                        best_conc_params,
                        method='Nelder-Mead',
                        options={'maxiter': 400, 'xatol': 1e-10, 'fatol': 1e-12}
                    )
                    conc_params_exact = res_conc_exact.x
                    conc_pts_exact = _make_concentric_points(conc_params_exact)
                    conc_score_exact = _score(conc_pts_exact)
                    
                    if conc_score_exact > best_score:
                        best_score = conc_score_exact
                        best_pts = conc_pts_exact.copy()
                except Exception:
                    pass

    # 4. Final High-Precision Polish
    # Multiple restarts from best and near-best points
    if time.time() < deadline - 10.0:
        # Collect candidates for final polish
        polish_candidates = [best_pts.copy()]
        for s, p in scored_inits[:3]:
            if not any(np.allclose(p, pc, atol=1e-5) for pc in polish_candidates):
                polish_candidates.append(p.copy())
        
        for pc in polish_candidates:
            if time.time() > deadline - 5.0:
                break
            try:
                res_final = minimize(
                    _neg_score,
                    pc.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                candidate_pts = res_final.x.reshape(-1, 2)
                candidate_score = _score(candidate_pts)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_pts = candidate_pts.copy()
            except Exception:
                pass
    
    # 5. Ultra-final micro-polish with very tight tolerances
    if time.time() < deadline - 3.0:
        try:
            res_ultra = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-14, 'fatol': 1e-16}
            )
            ultra_pts = res_ultra.x.reshape(-1, 2)
            ultra_score = _score(ultra_pts)
            if ultra_score > best_score:
                best_score = ultra_score
                best_pts = ultra_pts.copy()
        except Exception:
            pass
            
    return best_pts
