"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
N = 13

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import time
import itertools

def _min_triangle_area(pts):
    """Compute minimum triangle area among all triples, vectorized."""
    n = len(pts)
    if n < 3:
        return 0.0
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            v1 = pts[j] - pts[i]
            v2 = pts[j+1:] - pts[i]
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            areas = 0.5 * cross
            if len(areas) > 0:
                local_min = np.min(areas)
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

def _make_ring_config(params):
    """
    Construct 13 points from a ring configuration:
    12 points on outer ring + 1 center point.
    params: [r, theta0, d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11]
    where d_i are angular deviations from equally spaced positions.
    """
    r = params[0]
    theta0 = params[1]
    deviations = params[2:]
    
    base_angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    angles = base_angles + theta0 + deviations
    
    pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
    center = np.array([[0.0, 0.0]])
    
    return np.vstack([pts, center])

def _neg_ring_score(params):
    pts = _make_ring_config(params)
    return -_score(pts)

def _make_two_ring_config(params):
    """
    Construct 13 points from two rings:
    7 points on outer ring + 6 points on inner ring.
    params: [r1, r2, theta1, theta2, dev1_0..dev1_6, dev2_0..dev2_5]
    Total params: 2 + 2 + 7 + 6 = 17
    """
    r1, r2 = params[0], params[1]
    theta1, theta2 = params[2], params[3]
    dev1 = params[4:11]  # 7 deviations
    dev2 = params[11:17]  # 6 deviations
    
    angles1 = np.linspace(0, 2 * np.pi, 7, endpoint=False) + theta1 + dev1
    angles2 = np.linspace(0, 2 * np.pi, 6, endpoint=False) + theta2 + dev2
    
    pts1 = np.column_stack([r1 * np.cos(angles1), r1 * np.sin(angles1)])
    pts2 = np.column_stack([r2 * np.cos(angles2), r2 * np.sin(angles2)])
    
    return np.vstack([pts1, pts2])

def _neg_two_ring_score(params):
    pts = _make_two_ring_config(params)
    return -_score(pts)

def _make_hex_config(params):
    """
    Construct 13 points: 6 outer hexagon + 6 inner hexagon + 1 center.
    params: [r_out, r_in, theta_out, theta_in, cx, cy, dev_out_0..5, dev_in_0..5]
    Total: 6 + 12 = 18 params
    """
    r_out, r_in, th_out, th_in, cx, cy = params[:6]
    dev_out = params[6:12]
    dev_in = params[12:18]
    
    angles_out = np.linspace(0, 2 * np.pi, 6, endpoint=False) + th_out + dev_out
    angles_in = np.linspace(0, 2 * np.pi, 6, endpoint=False) + th_in + dev_in
    
    pts_out = np.column_stack([r_out * np.cos(angles_out), r_out * np.sin(angles_out)])
    pts_in = np.column_stack([r_in * np.cos(angles_in), r_in * np.sin(angles_in)])
    pts_c = np.array([[cx, cy]])
    
    return np.vstack([pts_out, pts_in, pts_c])

def _neg_hex_score(params):
    pts = _make_hex_config(params)
    return -_score(pts)

def _soft_min_triangle_area(pts, alpha=50.0):
    """Smooth approximation of the minimum triangle area."""
    n = len(pts)
    if n < 3:
        return 0.0
    log_sum = -np.inf
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            v1 = pts[j] - pts[i]
            v2 = pts[j+1:] - pts[i]
            cross = np.abs(v1[0] * v2[:, 1] - v1[1] * v2[:, 0])
            areas = 0.5 * cross
            if len(areas) > 0:
                min_a = np.min(areas)
                if min_a > 0:
                    shifted = -alpha * (areas - min_a)
                    log_sum_partial = np.log(np.sum(np.exp(shifted))) - alpha * min_a
                    if log_sum_partial > log_sum:
                        log_sum = log_sum_partial
    if log_sum == -np.inf:
        return 0.0
    return -log_sum / alpha

def _smooth_score(pts, alpha=50.0):
    """Smooth score surrogate."""
    n = len(pts)
    if n < 3:
        return 0.0
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    return _soft_min_triangle_area(pts, alpha) / hull_a

def _neg_smooth_score(params, alpha=50.0):
    pts = params.reshape(-1, 2)
    return -_smooth_score(pts, alpha)

def _neg_ring_smooth_score(params, alpha=50.0):
    pts = _make_ring_config(params)
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 1e10
    smooth_min = _soft_min_triangle_area(pts, alpha)
    return -(smooth_min / hull_a)

def _neg_two_ring_smooth_score(params, alpha=50.0):
    pts = _make_two_ring_config(params)
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 1e10
    smooth_min = _soft_min_triangle_area(pts, alpha)
    return -(smooth_min / hull_a)

def _neg_hex_smooth_score(params, alpha=50.0):
    pts = _make_hex_config(params)
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 1e10
    smooth_min = _soft_min_triangle_area(pts, alpha)
    return -(smooth_min / hull_a)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit
    
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
    
    # Hexagonal structure (6+6+1)
    for r_in in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85]:
        for th_in in [0.0, np.pi/6.0, np.pi/12.0]:
            params = [1.0, r_in, 0.0, th_in, 0.0, 0.0] + [0.0]*12
            pts = _make_hex_config(params)
            inits.append(pts)
    
    # Two-ring structure (7+6)
    for r2 in [0.4, 0.5, 0.6, 0.7, 0.8]:
        for th2 in [0.0, np.pi/6.0, np.pi/12.0]:
            params = [1.0, r2, 0.0, th2] + [0.0]*13
            pts = _make_two_ring_config(params)
            inits.append(pts)
    
    # Jittered regular polygon
    for jitter in [0.01, 0.05, 0.1, 0.15]:
        r = 1.0 + jitter * rng.randn(N)
        inits.append(np.column_stack([r * np.cos(angles), r * np.sin(angles)]))
        
    # Random points in unit disk
    for _ in range(10):
        r = rng.rand(N) ** 0.5
        theta = rng.rand(N) * 2 * np.pi
        inits.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
        
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
    
    # 2. Global optimization with differential evolution on full 26D space
    if time.time() < deadline - 40.0:
        try:
            bounds = [(-2, 2)] * (2 * N)
            de_result = differential_evolution(
                _neg_score,
                bounds,
                seed=42,
                maxiter=100,
                popsize=15,
                tol=1e-8,
                init='sobol',
                polish=False,
                workers=1,
                updating='immediate',
                seed=42
            )
            de_pts = de_result.x.reshape(-1, 2)
            de_score = _score(de_pts)
            if de_score > best_score:
                best_score = de_score
                best_pts = de_pts.copy()
        except Exception:
            pass
    
    # 3. Local refinement of top candidates
    top_candidates = []
    for s, p in scored_inits[:6]:
        top_candidates.append(p)
    top_candidates.append(best_pts)
    
    # Remove duplicates
    unique_candidates = []
    for c in top_candidates:
        if not any(np.allclose(c, u, atol=1e-5) for u in unique_candidates):
            unique_candidates.append(c)
    
    # Progressive annealing for each candidate
    alpha_schedule = [10.0, 30.0, 100.0, 300.0, 1000.0]
    
    for init_pts in unique_candidates:
        if time.time() > deadline - 30.0:
            break
            
        current_pts = init_pts.copy()
        current_score = _score(current_pts)
        
        # Progressive annealing with smooth surrogate
        for alpha in alpha_schedule:
            if time.time() > deadline - 25.0:
                break
            try:
                res_smooth = minimize(
                    lambda p, a=alpha: _neg_smooth_score(p, a),
                    current_pts.flatten(),
                    method='Powell',
                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12}
                )
                smooth_pts = res_smooth.x.reshape(-1, 2)
                smooth_score = _score(smooth_pts)
                if smooth_score > current_score:
                    current_pts = smooth_pts
                    current_score = smooth_score
            except Exception:
                pass
        
        # Final exact refinement with Nelder-Mead
        if time.time() > deadline - 15.0:
            continue
        try:
            res_exact = minimize(
                _neg_score,
                current_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-14}
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
    
    # 4. Symmetric optimization in reduced parameter spaces
    if time.time() < deadline - 25.0:
        # Ring configuration (12+1)
        ring_inits = [
            [1.0, 0.0] + [0.0]*11,
            [1.0, np.pi/12.0] + [0.0]*11,
            [1.0, np.pi/24.0] + [0.0]*11,
        ]
        for _ in range(5):
            ring_inits.append([
                1.0 + 0.05*rng.randn(),
                0.05*rng.randn(),
            ] + [0.05*rng.randn() for _ in range(11)])
        
        for ring_init in ring_inits:
            if time.time() > deadline - 20.0:
                break
            current_params = np.array(ring_init)
            current_score = _score(_make_ring_config(current_params))
            
            for alpha in [20.0, 50.0, 100.0, 300.0]:
                if time.time() > deadline - 15.0:
                    break
                try:
                    res = minimize(
                        lambda p, a=alpha: _neg_ring_smooth_score(p, a),
                        current_params,
                        method='L-BFGS-B',
                        options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    new_params = res.x
                    new_score = _score(_make_ring_config(new_params))
                    if new_score > current_score:
                        current_params = new_params
                        current_score = new_score
                except Exception:
                    pass
            
            if time.time() > deadline - 10.0:
                continue
            try:
                res_exact = minimize(
                    _neg_ring_score,
                    current_params,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                exact_params = res_exact.x
                exact_score = _score(_make_ring_config(exact_params))
                if exact_score > current_score:
                    current_params = exact_params
                    current_score = exact_score
            except Exception:
                pass
            
            if current_score > best_score:
                best_score = current_score
                best_pts = _make_ring_config(current_params).copy()
        
        # Two-ring configuration (7+6)
        if time.time() < deadline - 20.0:
            two_ring_inits = [
                [1.0, 0.6, 0.0, 0.0] + [0.0]*13,
                [1.0, 0.7, 0.0, np.pi/6.0] + [0.0]*13,
                [1.0, 0.5, np.pi/14.0, 0.0] + [0.0]*13,
            ]
            for _ in range(5):
                two_ring_inits.append([
                    1.0 + 0.05*rng.randn(),
                    0.6 + 0.1*rng.randn(),
                    0.05*rng.randn(),
                    0.05*rng.randn(),
                ] + [0.05*rng.randn() for _ in range(13)])
            
            for tr_init in two_ring_inits:
                if time.time() > deadline - 15.0:
                    break
                current_params = np.array(tr_init)
                current_score = _score(_make_two_ring_config(current_params))
                
                for alpha in [20.0, 50.0, 100.0]:
                    if time.time() > deadline - 12.0:
                        break
                    try:
                        res = minimize(
                            lambda p, a=alpha: _neg_two_ring_smooth_score(p, a),
                            current_params,
                            method='L-BFGS-B',
                            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                        )
                        new_params = res.x
                        new_score = _score(_make_two_ring_config(new_params))
                        if new_score > current_score:
                            current_params = new_params
                            current_score = new_score
                    except Exception:
                        pass
                
                if time.time() > deadline - 8.0:
                    continue
                try:
                    res_exact = minimize(
                        _neg_two_ring_score,
                        current_params,
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14}
                    )
                    exact_params = res_exact.x
                    exact_score = _score(_make_two_ring_config(exact_params))
                    if exact_score > current_score:
                        current_params = exact_params
                        current_score = exact_score
                except Exception:
                    pass
                
                if current_score > best_score:
                    best_score = current_score
                    best_pts = _make_two_ring_config(current_params).copy()
        
        # Hex configuration (6+6+1)
        if time.time() < deadline - 15.0:
            hex_inits = [
                [1.0, 0.5, 0.0, 0.0, 0.0, 0.0] + [0.0]*12,
                [1.0, 0.7, 0.0, np.pi/6.0, 0.0, 0.0] + [0.0]*12,
                [1.0, 0.6, 0.0, np.pi/12.0, 0.0, 0.0] + [0.0]*12,
            ]
            for _ in range(5):
                hex_inits.append([
                    1.0 + 0.05*rng.randn(),
                    0.5 + 0.15*rng.randn(),
                    0.05*rng.randn(),
                    0.05*rng.randn(),
                    0.02*rng.randn(),
                    0.02*rng.randn(),
                ] + [0.03*rng.randn() for _ in range(12)])
            
            for hex_init in hex_inits:
                if time.time() > deadline - 12.0:
                    break
                current_params = np.array(hex_init)
                current_score = _score(_make_hex_config(current_params))
                
                for alpha in [20.0, 50.0, 100.0]:
                    if time.time() > deadline - 10.0:
                        break
                    try:
                        res = minimize(
                            lambda p, a=alpha: _neg_hex_smooth_score(p, a),
                            current_params,
                            method='L-BFGS-B',
                            options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                        )
                        new_params = res.x
                        new_score = _score(_make_hex_config(new_params))
                        if new_score > current_score:
                            current_params = new_params
                            current_score = new_score
                    except Exception:
                        pass
                
                if time.time() > deadline - 7.0:
                    continue
                try:
                    res_exact = minimize(
                        _neg_hex_score,
                        current_params,
                        method='Nelder-Mead',
                        options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14}
                    )
                    exact_params = res_exact.x
                    exact_score = _score(_make_hex_config(exact_params))
                    if exact_score > current_score:
                        current_params = exact_params
                        current_score = exact_score
                except Exception:
                    pass
                
                if current_score > best_score:
                    best_score = current_score
                    best_pts = _make_hex_config(current_params).copy()
    
    # 5. Final High-Precision Polish
    if time.time() < deadline - 5.0:
        try:
            res_final = minimize(
                _neg_score,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-14, 'fatol': 1e-16}
            )
            candidate_pts = res_final.x.reshape(-1, 2)
            candidate_score = _score(candidate_pts)
            if candidate_score > best_score:
                best_score = candidate_score
                best_pts = candidate_pts.copy()
        except Exception:
            pass
            
    return best_pts
# EVOLVE-BLOCK-END
