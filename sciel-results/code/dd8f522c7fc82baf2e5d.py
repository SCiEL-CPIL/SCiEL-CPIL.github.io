"""Heilbronn convex regions: maximize min triangle area / convex hull area for 13 points."""
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import differential_evolution, minimize

N = 13
DEADLINE = 135.0  # safety margin before 150s kill

# Pre-compute all triples for fast vectorized scoring
# There are C(13, 3) = 286 triples
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

def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas vectorized."""
    pi = pts[_triple_i]
    pj = pts[_triple_j]
    pk = pts[_triple_k]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    return 0.5 * np.abs(cross)

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
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 0.0
    areas = _triangle_areas(pts)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_a

def _score_points(pts):
    """Compute min triangle area / convex hull area for given points."""
    n = len(pts)
    if n < 3:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    
    # Compute convex hull area
    try:
        hull = ConvexHull(pts)
        hull_area = hull.volume  # In 2D, volume is area
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    # Vectorized computation of all triangle areas
    pi = pts[_triple_i]
    pj = pts[_triple_j]
    pk = pts[_triple_k]
    
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    
    if min_area < 1e-15:
        return 0.0
    
    return min_area / hull_area

def _triangle_areas(pts):
    """Compute all C(n,3) triangle areas vectorized."""
    pi = pts[_triple_i]
    pj = pts[_triple_j]
    pk = pts[_triple_k]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    return 0.5 * np.abs(cross)

def _hull_area(pts):
    """Compute convex hull area."""
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 0.0

def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
    """Generate a regular n-gon."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack([x, y])

def _make_two_layer(inner_n, outer_n, r_inner=0.5, r_outer=1.0):
    """
    Create a two-layer configuration.
    inner_n points on a circle of radius r_inner.
    outer_n points on a circle of radius r_outer.
    Total points must be N.
    """
    if inner_n + outer_n != N:
        raise ValueError("inner_n + outer_n must equal N")
    
    pts = []
    if inner_n > 0:
        angles_in = np.linspace(0, 2 * np.pi, inner_n, endpoint=False)
        x_in = r_inner * np.cos(angles_in)
        y_in = r_inner * np.sin(angles_in)
        pts.append(np.column_stack([x_in, y_in]))
        
    if outer_n > 0:
        angles_out = np.linspace(0, 2 * np.pi, outer_n, endpoint=False)
        x_out = r_outer * np.cos(angles_out)
        y_out = r_outer * np.sin(angles_out)
        pts.append(np.column_stack([x_out, y_out]))
        
    return np.vstack(pts)

def _make_three_layer(center_n, mid_n, outer_n, r_mid=0.5, r_outer=1.0):
    """
    Create a three-layer configuration.
    center_n points at center (or small ring).
    mid_n points on circle radius r_mid.
    outer_n points on circle radius r_outer.
    """
    if center_n + mid_n + outer_n != N:
        raise ValueError("center_n + mid_n + outer_n must equal N")
        
    pts = []
    if center_n == 1:
        pts.append(np.array([[0.0, 0.0]]))
    elif center_n > 1:
        angles_c = np.linspace(0, 2 * np.pi, center_n, endpoint=False)
        r_c = 0.1
        x_c = r_c * np.cos(angles_c)
        y_c = r_c * np.sin(angles_c)
        pts.append(np.column_stack([x_c, y_c]))
        
    if mid_n > 0:
        angles_m = np.linspace(0, 2 * np.pi, mid_n, endpoint=False)
        x_m = r_mid * np.cos(angles_m)
        y_m = r_mid * np.sin(angles_m)
        pts.append(np.column_stack([x_m, y_m]))
        
    if outer_n > 0:
        angles_o = np.linspace(0, 2 * np.pi, outer_n, endpoint=False)
        x_o = r_outer * np.cos(angles_o)
        y_o = r_outer * np.sin(angles_o)
        pts.append(np.column_stack([x_o, y_o]))
        
    return np.vstack(pts)

def _logsumexp_surrogate(pts, beta=50.0):
    """
    Smooth surrogate: -logsumexp(-beta * areas) approximates min(areas) for large beta.
    This gives a differentiable objective that we can optimize with gradient methods.
    """
    areas = _triangle_areas(pts)
    if np.any(areas < 1e-15):
        return 0.0
    # -logsumexp(-beta * areas) / beta approximates min(areas)
    # Use stable logsumexp
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    if max_scaled > 700:  # overflow guard
        return areas[np.argmin(areas)]
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -log_sum / beta

def _neg_score(params):
    """Negative score for minimization. params is 1D array of length 2*N."""
    pts = params.reshape(N, 2)
    return -_score_points(pts)

def _neg_surrogate(params, beta=50.0):
    """Negative log-sum-exp surrogate for minimization."""
    pts = params.reshape(N, 2)
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 1.0
    surrogate_val = _logsumexp_surrogate(pts, beta)
    return -surrogate_val / hull_a

def _soft_min_score(params, beta=30.0):
    """
    Soft-min objective using log-sum-exp:
    f(x) = -(1/beta) * log(sum(exp(-beta * a_i / hull)))
    approximates -min(a_i) / hull for large beta.
    """
    pts = params.reshape(N, 2)
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 1.0
    areas = _triangle_areas(pts)
    if np.any(areas < 1e-15):
        return 1.0
    scaled = -beta * areas / hull_a
    max_s = np.max(scaled)
    if max_s > 700:
        min_a = np.min(areas)
        return -min_a / hull_a
    log_sum = max_s + np.log(np.sum(np.exp(scaled - max_s)))
    return -(1.0 / beta) * log_sum

def _neg_soft_score(params, beta=30.0):
    return _soft_min_score(params, beta)

def _neg_hard_score(params):
    return _neg_score(params)

def _build_ring_config(params, n_inner, n_outer, r_inner_base, r_outer_base, rot_inner_base, rot_outer_base):
    """
    Build points from ring parameters.
    params: [r_inner_scale, r_outer_scale, rot_inner, rot_outer] (4 params for 2-ring)
    or [r1, r2, r3, rot1, rot2, rot3] (6 params for 3-ring)
    Returns (N, 2) array.
    """
    pts = np.zeros((N, 2))
    if n_inner == 1:
        # Single center point
        pts[0] = [0.0, 0.0]
        if n_outer == N - 1:
            r = params[0] * r_outer_base
            rot = params[1] + rot_outer_base
            angles = rot + np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
            pts[1:, 0] = r * np.cos(angles)
            pts[1:, 1] = r * np.sin(angles)
        return pts
    elif n_inner + n_outer == N and n_inner > 1:
        # Two rings: n_inner on inner circle, n_outer on outer circle
        r_in = params[0] * r_inner_base
        r_out = params[1] * r_outer_base
        rot_in = params[2] + rot_inner_base
        rot_out = params[3] + rot_outer_base
        angles_in = rot_in + np.linspace(0, 2 * np.pi, n_inner, endpoint=False)
        angles_out = rot_out + np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
        pts[:n_inner, 0] = r_in * np.cos(angles_in)
        pts[:n_inner, 1] = r_in * np.sin(angles_in)
        pts[n_inner:, 0] = r_out * np.cos(angles_out)
        pts[n_inner:, 1] = r_out * np.sin(angles_out)
        return pts
    return pts

def _neg_ring_score(params, n_inner, n_outer, r_inner_base, r_outer_base, rot_inner_base, rot_outer_base):
    """Negative score for ring-structured optimization."""
    pts = _build_ring_config(params, n_inner, n_outer, r_inner_base, r_outer_base, rot_inner_base, rot_outer_base)
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 1.0
    areas = _triangle_areas(pts)
    min_a = np.min(areas)
    if min_a < 1e-15:
        return 1.0
    return -min_a / hull_a

def _neg_ring_surrogate(params, beta, n_inner, n_outer, r_inner_base, r_outer_base, rot_inner_base, rot_outer_base):
    """Smooth surrogate for ring optimization."""
    pts = _build_ring_config(params, n_inner, n_outer, r_inner_base, r_outer_base, rot_inner_base, rot_outer_base)
    hull_a = _hull_area(pts)
    if hull_a <= 1e-12:
        return 1.0
    areas = _triangle_areas(pts)
    if np.any(areas < 1e-15):
        return 1.0
    scaled = -beta * areas
    max_s = np.max(scaled)
    if max_s > 700:
        min_a = np.min(areas)
        return -min_a / hull_a
    log_sum = max_s + np.log(np.sum(np.exp(scaled - max_s)))
    surrogate = -log_sum / beta
    return -surrogate / hull_a

def _optimize_ring_config(t_start, n_inner, n_outer, r_inner_base, r_outer_base, rot_inner_base, rot_outer_base, deadline_margin=10.0):
    """
    Optimize a specific ring configuration with few parameters.
    For 1-center + (N-1)-outer: params = [r_outer_scale, rot_outer] (2 params)
    For two rings: params = [r_inner_scale, r_outer_scale, rot_inner, rot_outer] (4 params)
    """
    best_score = -np.inf
    best_params = None
    
    if n_inner == 1:
        # 1 center + (N-1) outer: 2 params
        n_params = 2
        bounds = [(0.3, 2.0), (0.0, 2 * np.pi)]
        init = np.array([1.0, 0.0])
    else:
        # Two rings: 4 params
        n_params = 4
        bounds = [(0.1, 1.5), (0.5, 2.5), (0.0, 2 * np.pi), (0.0, 2 * np.pi)]
        init = np.array([1.0, 1.0, 0.0, 0.0])
    
    def neg_obj(params):
        return _neg_ring_score(params, n_inner, n_outer, r_inner_base, r_outer_base, rot_inner_base, rot_outer_base)
    
    def neg_surr(params, beta=20.0):
        return _neg_ring_surrogate(params, beta, n_inner, n_outer, r_inner_base, r_outer_base, rot_inner_base, rot_outer_base)
    
    # Phase 1: Differential evolution (small search space, fast)
    try:
        de_result = differential_evolution(
            neg_obj,
            bounds,
            seed=42,
            maxiter=40,
            popsize=15,
            tol=1e-6,
            mutation=(0.4, 0.9),
            recombination=0.7,
            polish=True,
            init='sobol',
            updating='deferred',
            workers=1
        )
        if de_result.fun < -best_score:
            best_score = -de_result.fun
            best_params = de_result.x.copy()
    except Exception:
        pass
    
    if best_params is None:
        best_params = init.copy()
    
    # Phase 2: Local refinement with multiple starts
    rng = np.random.RandomState(7)
    starts = [best_params.copy(), init.copy()]
    for _ in range(4):
        pert = rng.normal(0, 0.15, size=n_params)
        start = best_params + pert
        start = np.clip(start, [b[0] for b in bounds], [b[1] for b in bounds])
        starts.append(start)
    
    for start in starts:
        if time.time() - t_start > deadline_margin:
            break
        try:
            res = minimize(
                neg_obj, start,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 400, 'maxfun': 8000}
            )
            s = -res.fun
            if s > best_score:
                best_score = s
                best_params = res.x.copy()
        except Exception:
            pass
    
    # Phase 3: Surrogate refinement
    for beta in [10.0, 25.0]:
        if time.time() - t_start > deadline_margin:
            break
        try:
            res_s = minimize(
                lambda p: neg_surr(p, beta=beta), best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'maxfun': 4000}
            )
            s = _score_points(_build_ring_config(res_s.x, n_inner, n_outer, r_inner_base, r_outer_base, rot_inner_base, rot_outer_base))
            if s > best_score:
                best_score = s
                best_params = res_s.x.copy()
        except Exception:
            pass
    
    pts = _build_ring_config(best_params, n_inner, n_outer, r_inner_base, r_outer_base, rot_inner_base, rot_outer_base)
    return pts, best_score

def _equalize_active_triangles(pts, t_start, max_iters=200, tolerance=1e-8):
    """
    Equalize the areas of the 'active' (near-minimum) triangles.
    """
    best_pts = pts.copy()
    best_score = _score_points(best_pts)
    current_pts = pts.copy()
    
    for it in range(max_iters):
        elapsed = time.time() - t_start
        if elapsed > DEADLINE - 15.0:
            break
        
        areas = _triangle_areas(current_pts)
        hull_a = _hull_area(current_pts)
        if hull_a < 1e-12:
            break
        
        min_area = np.min(areas)
        if min_area < 1e-15:
            break
        
        threshold = min_area * 1.05
        active_mask = areas <= threshold
        active_idx = np.where(active_mask)[0]
        
        if len(active_idx) == 0:
            break
        
        displacement = np.zeros_like(current_pts)
        
        for idx in active_idx:
            i1 = _triple_i[idx]
            i2 = _triple_j[idx]
            i3 = _triple_k[idx]
            
            p1 = current_pts[i1]
            p2 = current_pts[i2]
            p3 = current_pts[i3]
            
            v12 = p2 - p1
            v13 = p3 - p1
            cross_val = v12[0] * v13[1] - v12[1] * v13[0]
            
            if abs(cross_val) < 1e-15:
                continue
            
            sign = np.sign(cross_val)
            
            mid_23 = (p2 + p3) / 2.0
            dir_1 = p1 - mid_23
            
            mid_13 = (p1 + p3) / 2.0
            dir_2 = p2 - mid_13
            
            mid_12 = (p1 + p2) / 2.0
            dir_3 = p3 - mid_12
            
            n1 = np.linalg.norm(dir_1)
            n2 = np.linalg.norm(dir_2)
            n3 = np.linalg.norm(dir_3)
            
            if n1 > 1e-10:
                displacement[i1] += sign * dir_1 / n1
            if n2 > 1e-10:
                displacement[i2] += sign * dir_2 / n2
            if n3 > 1e-10:
                displacement[i3] += sign * dir_3 / n3
        
        disp_norm = np.linalg.norm(displacement)
        if disp_norm < 1e-15:
            break
        
        step_size = 0.01
        proposed_pts = current_pts + step_size * displacement / disp_norm
        proposed_pts = np.clip(proposed_pts, -3.0, 3.0)
        
        proposed_score = _score_points(proposed_pts)
        current_score = _score_points(current_pts)
        
        if proposed_score > current_score - 1e-10:
            current_pts = proposed_pts
            if proposed_score > best_score:
                best_score = proposed_score
                best_pts = current_pts.copy()
        else:
            step_size *= 0.5
            if step_size < 1e-8:
                break
    
    return best_pts, best_score

def construct():
    t_start = time.time()
    
    best_score = -np.inf
    best_pts = None
    
    def time_remaining():
        return (DEADLINE - 5.0) - (time.time() - t_start)
    
    # === Phase 1: Multi-ring symmetry-constrained optimization ===
    # Explore several structural families with very few parameters each
    ring_configs = [
        # (n_inner, n_outer, r_inner_base, r_outer_base, rot_inner_base, rot_outer_base)
        (1, 12, 0.0, 1.0, 0.0, 0.0),           # 1 center + 12 outer
        (1, 12, 0.0, 1.0, 0.0, np.pi/12),      # 1 center + 12 outer, rotated
        (1, 12, 0.0, 1.0, 0.0, np.pi/6),       # 1 center + 12 outer, more rotation
        (5, 8, 0.5, 1.0, 0.0, 0.0),            # 5 inner + 8 outer
        (5, 8, 0.5, 1.0, 0.0, np.pi/8),        # 5 inner + 8 outer, rotated
        (5, 8, 0.5, 1.0, np.pi/5, 0.0),        # 5 inner rotated + 8 outer
        (4, 9, 0.4, 1.0, 0.0, 0.0),            # 4 inner + 9 outer
        (4, 9, 0.4, 1.0, 0.0, np.pi/9),        # 4 inner + 9 outer, rotated
        (3, 10, 0.4, 1.0, 0.0, 0.0),           # 3 inner + 10 outer
        (3, 10, 0.4, 1.0, 0.0, np.pi/10),      # 3 inner + 10 outer, rotated
        (6, 7, 0.5, 1.0, 0.0, 0.0),            # 6 inner + 7 outer
        (6, 7, 0.5, 1.0, 0.0, np.pi/7),        # 6 inner + 7 outer, rotated
        (2, 11, 0.3, 1.0, 0.0, 0.0),           # 2 inner + 11 outer
        (2, 11, 0.3, 1.0, 0.0, np.pi/11),      # 2 inner + 11 outer, rotated
        (7, 6, 0.5, 1.0, 0.0, 0.0),            # 7 inner + 6 outer
        (7, 6, 0.5, 1.0, 0.0, np.pi/6),        # 7 inner + 6 outer, rotated
    ]
    
    # Allocate time per config
    n_configs = len(ring_configs)
    time_per_config = min(8.0, (DEADLINE - 50.0) / n_configs)
    
    for cfg in ring_configs:
        if time_remaining() < 20.0:
            break
        n_inner, n_outer, r_in, r_out, rot_in, rot_out = cfg
        margin = min(time_per_config, time_remaining() - 15.0)
        if margin < 3.0:
            break
        try:
            cfg_pts, cfg_score = _optimize_ring_config(
                t_start, n_inner, n_outer, r_in, r_out, rot_in, rot_out, deadline_margin=margin
            )
            if cfg_pts is not None and cfg_score > best_score:
                best_score = cfg_score
                best_pts = cfg_pts
        except Exception:
            pass
    
    # === Phase 2: Free-space local refinement from best ring solution ===
    bounds = [(-2.0, 2.0)] * (2 * N)
    
    if best_pts is not None:
        # Try L-BFGS on full 26-dim from best ring solution
        if time_remaining() > 20.0:
            try:
                res = minimize(
                    _neg_score, best_pts.flatten(),
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 300, 'maxfun': 10000}
                )
                s = _score_points(res.x.reshape(N, 2))
                if s > best_score:
                    best_score = s
                    best_pts = res.x.reshape(N, 2)
            except Exception:
                pass
        
        # Surrogate refinement
        if time_remaining() > 15.0:
            for beta in [15.0, 30.0]:
                if time_remaining() < 8.0:
                    break
                try:
                    def neg_surr_full(p, b=beta):
                        pts = p.reshape(N, 2)
                        hull_a = _hull_area(pts)
                        if hull_a <= 1e-12:
                            return 1.0
                        areas = _triangle_areas(pts)
                        if np.any(areas < 1e-15):
                            return 1.0
                        scaled = -b * areas
                        max_s = np.max(scaled)
                        if max_s > 700:
                            return -np.min(areas) / hull_a
                        log_sum = max_s + np.log(np.sum(np.exp(scaled - max_s)))
                        return -(-log_sum / b) / hull_a
                    
                    res_s = minimize(
                        neg_surr_full, best_pts.flatten(),
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={'maxiter': 200, 'maxfun': 5000}
                    )
                    s = _score_points(res_s.x.reshape(N, 2))
                    if s > best_score:
                        best_score = s
                        best_pts = res_s.x.reshape(N, 2)
                except Exception:
                    pass
    
    # === Phase 3: Equalize active triangles ===
    if time_remaining() > 10.0 and best_pts is not None:
        eq_pts, eq_score = _equalize_active_triangles(best_pts, t_start, max_iters=200)
        if eq_score > best_score:
            best_score = eq_score
            best_pts = eq_pts
    
    # === Phase 4: Final local polish ===
    if time_remaining() > 5.0 and best_pts is not None:
        try:
            res_final = minimize(
                _neg_score, best_pts.flatten(),
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100, 'maxfun': 3000}
            )
            s = _score_points(res_final.x.reshape(N, 2))
            if s > best_score:
                best_score = s
                best_pts = res_final.x.reshape(N, 2)
        except Exception:
            pass
    
    # === Phase 5: Final equalization ===
    if time_remaining() > 3.0 and best_pts is not None:
        eq2_pts, eq2_score = _equalize_active_triangles(best_pts, t_start, max_iters=80)
        if eq2_score > best_score:
            best_score = eq2_score
            best_pts = eq2_pts
    
    # Safety: ensure we return a valid point set
    if best_pts is None:
        best_pts = _regular_polygon(N)
    
    return best_pts
