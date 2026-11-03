"""
Heilbronn convex regions n=14: maximize min triangle area / convex hull area.

Combined strategy:
  - Primary: centrally symmetric polygon (7 antipodal pairs), 14D param space
  - Secondary: B-spline boundary (7 control points), ring (2x7)
  - Soft-min surrogate refinement via log-sum-exp for smooth gradients
  - Multi-phase: NM on parameter spaces -> Cartesian NM -> soft-min L-BFGS-B ->
    hard NM polish -> Powell -> coordinate descent
  - Tight deadline guard (145s) for deeper search within 150s budget
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
from scipy.interpolate import CubicSpline
import itertools

N = 14

# Pre-compute all triangle index combinations
tri_indices = np.array(list(itertools.combinations(range(N), 3)))  # shape (364, 3)


def _min_triangle_area(points):
    """Minimum triangle area over all triples, vectorized."""
    p = points
    pi = p[tri_indices[:, 0]]
    pj = p[tri_indices[:, 1]]
    pk = p[tri_indices[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _all_triangle_areas(points):
    pi = points[tri_indices[:, 0]]
    pj = points[tri_indices[:, 1]]
    pk = points[tri_indices[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross)


def _hull_area(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0


def _score(points):
    hull_area = _hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area(points)
    return min_area / hull_area


def _objective_hard(flat):
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    min_area = _min_triangle_area(pts)
    if min_area < 1e-12:
        return 1e10
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / min_area


def _objective_soft(flat, beta=100.0):
    """Smooth soft-min objective: hull_area / soft_min(triangle areas)."""
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    areas = _all_triangle_areas(pts)
    if np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min = -log_sum / beta
    if soft_min < 1e-12:
        return 1e10
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min


# ---------- Centrally symmetric polygon parameterization (PRIMARY) ----------
def _decode_sym(params):
    """
    Decode 14-dim params [theta_0..6, r_0..6] to 14 centrally symmetric points.
    Points: v_i = r_i*(cos theta_i, sin theta_i), and -v_i for i=0..6.
    """
    theta = params[0:7]
    r = params[7:14]
    r = np.maximum(r, 0.01)
    
    x1 = r * np.cos(theta)
    y1 = r * np.sin(theta)
    x2 = -x1
    y2 = -y1
    
    pts = np.column_stack([np.concatenate([x1, x2]), np.concatenate([y1, y2])])
    return pts


def _objective_sym(params):
    pts = _decode_sym(params)
    return _objective_hard(pts.flatten())


def _objective_sym_soft(params, beta=100.0):
    pts = _decode_sym(params)
    return _objective_soft(pts.flatten(), beta)


def _generate_sym_inits():
    """Generate diverse initial parameter vectors for centrally symmetric optimization."""
    inits = []
    rng = np.random.default_rng(42)
    
    # 1. Regular 14-gon (centrally symmetric): 7 equally spaced angles
    base_angles = np.linspace(0, np.pi, 7, endpoint=False)
    inits.append(np.concatenate([base_angles, np.ones(7)])).copy()
    
    # 2. Rotated regular
    for rot in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.3, 1.6]:
        a = base_angles + rot
        inits.append(np.concatenate([a, np.ones(7)]).copy())
    
    # 3. Elliptical distortions
    for rx, ry in [(1.1, 0.9), (0.9, 1.1), (1.2, 0.85), (0.85, 1.2), (1.15, 0.87)]:
        a = base_angles
        r = np.sqrt(rx**2 * np.cos(a)**2 + ry**2 * np.sin(a)**2)
        inits.append(np.concatenate([a, r]).copy())
    
    # 4. Varying radii patterns
    for seed in range(20):
        rs = np.random.default_rng(seed * 13 + 7)
        a = base_angles + rs.normal(0, 0.05, 7)
        r = 1.0 + rs.normal(0, 0.15, 7)
        r = np.maximum(r, 0.5)
        inits.append(np.concatenate([a, r]).copy())
    
    # 5. Alternating radii
    for r_pat in [
        [1.0, 0.85, 1.0, 0.85, 1.0, 0.85, 1.0],
        [1.0, 0.75, 0.95, 0.75, 1.0, 0.75, 0.95],
        [1.1, 0.9, 1.05, 0.9, 1.1, 0.9, 1.05],
    ]:
        a = base_angles
        inits.append(np.concatenate([a, np.array(r_pat)]).copy())
    
    # 6. Random convex-ish
    for seed in range(20):
        rs = np.random.default_rng(seed * 23 + 11)
        a = np.sort(rs.uniform(0, np.pi, 7))
        r = rs.uniform(0.7, 1.3, 7)
        inits.append(np.concatenate([a, r]).copy())
    
    # 7. Larger perturbations
    for seed in range(15):
        rs = np.random.default_rng(seed * 31 + 17)
        a = base_angles + rs.normal(0, 0.3, 7)
        r = 1.0 + rs.normal(0, 0.3, 7)
        r = np.maximum(r, 0.4)
        inits.append(np.concatenate([a, r]).copy())
    
    # 8. Jittered regular with various jitter levels
    for js in [0.05, 0.1, 0.15, 0.2, 0.3]:
        a = base_angles + rng.normal(0, js, 7)
        r = 1.0 + rng.normal(0, js * 0.5, 7)
        r = np.maximum(r, 0.5)
        inits.append(np.concatenate([a, r]).copy())
    
    return inits


# ---------- B-spline convex boundary parameterization (SECONDARY) ----------
def _decode_bspline(ctrl_params):
    """Decode 14-dim control point parameters to 14 points on a B-spline curve."""
    ctrl = ctrl_params.reshape(7, 2)
    
    try:
        ctrl_closed = np.vstack([ctrl, ctrl[0]])
        t = np.arange(8)
        cs_x = CubicSpline(t, ctrl_closed[:, 0], bc_type='periodic')
        cs_y = CubicSpline(t, ctrl_closed[:, 1], bc_type='periodic')
        
        t_fine = np.linspace(0, 7, 2000)
        x_fine = cs_x(t_fine)
        y_fine = cs_y(t_fine)
        
        dx = np.diff(x_fine)
        dy = np.diff(y_fine)
        ds = np.sqrt(dx**2 + dy**2)
        s = np.concatenate([[0], np.cumsum(ds)])
        total_s = s[-1]
        
        if total_s < 1e-10:
            raise ValueError("degenerate curve")
        
        s_targets = np.linspace(0, total_s, N, endpoint=False)
        x_pts = np.interp(s_targets, s, x_fine)
        y_pts = np.interp(s_targets, s, y_fine)
        
        pts = np.column_stack([x_pts, y_pts])
        pts -= pts.mean(axis=0)
        return pts
        
    except Exception:
        ctrl_centered = ctrl - ctrl.mean(axis=0)
        pts = np.vstack([ctrl_centered, -ctrl_centered[::-1]])
        return pts


def _objective_bspline(ctrl_params):
    pts = _decode_bspline(ctrl_params)
    return _objective_hard(pts.flatten())


def _objective_bspline_soft(ctrl_params, beta=100.0):
    pts = _decode_bspline(ctrl_params)
    return _objective_soft(pts.flatten(), beta)


def _generate_bspline_inits():
    """Generate diverse initial control point configurations for B-spline."""
    inits = []
    rng = np.random.default_rng(42)
    
    # Regular heptagon
    angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    ctrl = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(ctrl.flatten().copy())
    
    # Elliptical
    for rx, ry in [(1.2, 0.8), (0.8, 1.2), (1.1, 0.9), (0.9, 1.1)]:
        ctrl = np.column_stack([rx * np.cos(angles), ry * np.sin(angles)])
        inits.append(ctrl.flatten().copy())
    
    # Perturbed
    for js in [0.05, 0.1, 0.15, 0.2]:
        ctrl = np.column_stack([np.cos(angles), np.sin(angles)])
        ctrl += rng.normal(0, js, size=(7, 2))
        inits.append(ctrl.flatten().copy())
    
    # Random convex-ish
    for seed in range(15):
        rs = np.random.default_rng(seed * 13 + 7)
        angles_s = np.sort(rs.uniform(0, 2 * np.pi, 7))
        radii = rs.uniform(0.6, 1.4, 7)
        ctrl = np.column_stack([radii * np.cos(angles_s), radii * np.sin(angles_s)])
        inits.append(ctrl.flatten().copy())
    
    # Larger perturbations
    for seed in range(10):
        rs = np.random.default_rng(seed * 31 + 17)
        angles_s = np.sort(rs.uniform(0, 2 * np.pi, 7))
        radii = rs.uniform(0.4, 1.6, 7)
        ctrl = np.column_stack([radii * np.cos(angles_s), radii * np.sin(angles_s)])
        inits.append(ctrl.flatten().copy())
    
    return inits


# ---------- Ring parameterization (TERTIARY) ----------
def _decode_ring2(params):
    """Decode 18-dim ring params to (N,2) points."""
    r1 = params[0]
    r2 = params[1]
    phi1 = params[2]
    phi2 = params[3]
    d1 = params[4:11]
    d2 = params[11:18]
    
    base_angles1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    base_angles2 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    
    angles1 = phi1 + base_angles1 + d1
    angles2 = phi2 + base_angles2 + d2
    
    x1 = r1 * np.cos(angles1)
    y1 = r1 * np.sin(angles1)
    x2 = r2 * np.cos(angles2)
    y2 = r2 * np.sin(angles2)
    
    return np.column_stack([np.concatenate([x1, x2]), np.concatenate([y1, y2])])


def _objective_ring2(params):
    pts = _decode_ring2(params)
    return _objective_hard(pts.flatten())


def _objective_ring2_soft(params, beta=100.0):
    pts = _decode_ring2(params)
    return _objective_soft(pts.flatten(), beta)


def _make_ring2_inits():
    """Generate diverse initial ring-2 parameter vectors."""
    inits = []
    rng = np.random.default_rng(42)
    
    for r1 in [0.3, 0.5, 0.6, 0.7, 0.8, 0.85]:
        for phi2 in [np.pi / 7, 0.0, np.pi / 14]:
            params = np.zeros(18)
            params[0] = r1
            params[1] = 1.0
            params[2] = 0.0
            params[3] = phi2
            inits.append(params)
    
    for js in [0.05, 0.1, 0.15]:
        params = np.zeros(18)
        params[0] = 0.6
        params[1] = 1.0
        params[3] = np.pi / 7
        params[4:18] = rng.normal(0, js, size=14)
        inits.append(params)
    
    for seed in range(10):
        rs = np.random.default_rng(seed * 13 + 7)
        params = np.zeros(18)
        params[0] = rs.uniform(0.3, 0.8)
        params[1] = rs.uniform(0.9, 1.3)
        params[2] = rs.uniform(-0.2, 0.2)
        params[3] = rs.uniform(0.0, 2 * np.pi)
        params[4:18] = rs.normal(0, 0.1, size=14)
        inits.append(params)
    
    return inits


# ---------- Optimizer helpers ----------

def _nm(x0, deadline, obj, max_iter=2000):
    if time.time() > deadline - 3:
        return None
    try:
        res = minimize(obj, x0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
        return res
    except Exception:
        return None


def _powell(x0, deadline, obj, max_iter=1000):
    if time.time() > deadline - 3:
        return None
    try:
        res = minimize(obj, x0, method='Powell',
                       options={'maxiter': max_iter, 'xtol': 1e-12, 'ftol': 1e-14})
        return res
    except Exception:
        return None


def _lbfgs(x0, deadline, obj, max_iter=3000):
    if time.time() > deadline - 5:
        return None
    try:
        res = minimize(obj, x0, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-11,
                                'maxls': 50})
        return res
    except Exception:
        return None


def _normalize_unit_area(points):
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area > 1e-10:
            scale = np.sqrt(1.0 / hull_area)
            return points * scale
    except Exception:
        pass
    xs = points[:, 0]
    ys = points[:, 1]
    w = xs.max() - xs.min()
    h = ys.max() - ys.min()
    if w > 1e-10 and h > 1e-10:
        scale = 1.0 / np.sqrt(w * h)
        return points * scale
    return points


def _make_cartesian_inits():
    """Generate diverse initial point configurations (Cartesian)."""
    inits = []
    rng = np.random.default_rng(42)
    
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    inits.append(np.column_stack([np.cos(angles), np.sin(angles)]).copy())
    
    # Ring structures
    for r1 in [0.3, 0.5, 0.6, 0.7, 0.85]:
        a1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
        pts = np.vstack([
            r1 * np.column_stack([np.cos(a1), np.sin(a1)]),
            np.column_stack([np.cos(a2), np.sin(a2)])
        ])
        inits.append(pts.copy())
    
    # Random
    for seed in range(15):
        rs = np.random.default_rng(seed * 37 + 19)
        pts = rs.uniform(-1, 1, size=(N, 2))
        inits.append(pts)
    
    # Perturbed regular
    base = np.column_stack([np.cos(angles), np.sin(angles)])
    for js in [0.05, 0.1, 0.2, 0.3]:
        inits.append((base + rng.normal(0, js, size=(N, 2))).copy())
    
    # Elliptical
    for rx, ry in [(1.0, 0.6), (1.0, 0.8), (1.2, 0.7), (0.8, 1.2), (1.1, 0.9)]:
        a = np.linspace(0, 2 * np.pi, N, endpoint=False)
        inits.append(np.column_stack([rx * np.cos(a), ry * np.sin(a)]).copy())
    
    return inits


def _coordinate_descent(pts, deadline, max_rounds=25):
    """Cyclic coordinate descent on the hard objective."""
    pts = pts.copy()
    n = len(pts)
    step0 = 0.05
    for rnd in range(max_rounds):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(n):
            if time.time() > deadline - 5:
                break
            for coord in range(2):
                orig = pts[i, coord]
                step = step0
                for _ in range(25):
                    pts[i, coord] = orig + step
                    s_plus = _score(pts)
                    pts[i, coord] = orig - step
                    s_minus = _score(pts)
                    pts[i, coord] = orig
                    if s_plus > s_minus:
                        pts[i, coord] = orig + step
                    elif s_minus > s_plus:
                        pts[i, coord] = orig - step
                    else:
                        break
                    improved = True
                    step *= 0.5
                    if step < 1e-9:
                        break
        if not improved:
            break
    return pts


def _soft_min_refinement(pts, deadline, beta_start=50.0, beta_end=300.0, n_beta=5):
    """
    Refine points using soft-min surrogate with increasing beta.
    This helps escape flat regions of the hard min objective.
    """
    current_pts = pts.copy()
    betas = np.linspace(beta_start, beta_end, n_beta)
    
    for beta in betas:
        if time.time() > deadline - 10:
            break
        
        # Convert to Cartesian flat
        flat0 = current_pts.flatten()
        
        # L-BFGS-B on soft objective
        res = _lbfgs(flat0, deadline, lambda x: _objective_soft(x, beta=beta), max_iter=2000)
        if res is not None and res.fun < 1e9:
            new_pts = res.x.reshape(N, 2)
            if np.all(np.isfinite(new_pts)):
                new_score = _score(new_pts)
                cur_score = _score(current_pts)
                if new_score > cur_score:
                    current_pts = new_pts
    
    return current_pts


def _high_precision_polish(pts, deadline):
    """Final high-precision polish with very tight tolerances."""
    if time.time() > deadline - 8:
        return pts
    
    flat = pts.flatten()
    best_pts = pts.copy()
    best_obj = _objective_hard(flat)
    
    # NM with very tight tolerance
    if time.time() < deadline - 6:
        res = minimize(_objective_hard, flat, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
        if res is not None and res.fun < best_obj:
            best_pts = res.x.reshape(N, 2)
            best_obj = res.fun
    
    # Powell with tight tolerance
    if time.time() < deadline - 4:
        res = minimize(_objective_hard, best_pts.flatten(), method='Powell',
                       options={'maxiter': 2000, 'xtol': 1e-14, 'ftol': 1e-16})
        if res is not None and res.fun < best_obj:
            best_pts = res.x.reshape(N, 2)
            best_obj = res.fun
    
    return best_pts


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    best_score = 0.0
    best_pts = None
    best_obj = float('inf')
    
    def update_best(pts):
        nonlocal best_score, best_pts, best_obj
        if pts is None:
            return
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, 2) or not np.all(np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
            best_obj = _objective_hard(pts.flatten())
    
    # Phase 1: Centrally symmetric NM (PRIMARY - most structured)
    sym_inits = _generate_sym_inits()
    
    for i, p0 in enumerate(sym_inits):
        if time.time() > deadline - 50:
            break
        res = _nm(p0, deadline, _objective_sym, max_iter=2000)
        if res is not None and res.fun < best_obj:
            pts = _decode_sym(res.x)
            update_best(pts)
    
    # Phase 2: B-spline NM (SECONDARY)
    bspline_inits = _generate_bspline_inits()
    
    for i, p0 in enumerate(bspline_inits):
        if time.time() > deadline - 40:
            break
        res = _nm(p0, deadline, _objective_bspline, max_iter=1500)
        if res is not None and res.fun < best_obj:
            pts = _decode_bspline(res.x)
            update_best(pts)
    
    # Phase 3: Ring NM (TERTIARY)
    ring_inits = _make_ring2_inits()
    
    for i, p0 in enumerate(ring_inits):
        if time.time() > deadline - 35:
            break
        res = _nm(p0, deadline, _objective_ring2, max_iter=1200)
        if res is not None and res.fun < best_obj:
            pts = _decode_ring2(res.x)
            update_best(pts)
    
    # Phase 4: Cartesian NM from best found
    if best_pts is not None and time.time() < deadline - 30:
        res = _nm(best_pts.flatten(), deadline, _objective_hard, max_iter=3000)
        if res is not None and res.fun < best_obj:
            update_best(res.x.reshape(N, 2))
    
    # Phase 5: Soft-min surrogate refinement (smooth gradient escape)
    if best_pts is not None and time.time() < deadline - 20:
        refined = _soft_min_refinement(best_pts, deadline, beta_start=50.0, beta_end=300.0, n_beta=5)
        update_best(refined)
        
        # After soft refinement, re-polish with hard NM
        if time.time() < deadline - 15:
            res = _nm(best_pts.flatten(), deadline, _objective_hard, max_iter=3000)
            if res is not None and res.fun < best_obj:
                update_best(res.x.reshape(N, 2))
    
    # Phase 6: Differential Evolution (Global Search)
    if time.time() < deadline - 15:
        try:
            bounds = [(-2.0, 2.0)] * (N * 2)
            de_res = differential_evolution(
                _objective_hard, bounds,
                seed=42, maxiter=50, tol=1e-9,
                popsize=15, mutation=(0.4, 1.0), recombination=0.7,
                polish=False, update_interval=1,
                init='sobol'
            )
            if de_res.fun < best_obj:
                update_best(de_res.x.reshape(N, 2))
        except Exception:
            pass
    
    # Phase 7: Coordinate descent refinement
    if best_pts is not None and time.time() < deadline - 12:
        cd_pts = _coordinate_descent(best_pts, deadline, max_rounds=25)
        update_best(cd_pts)
    
    # Phase 8: Final high-precision polish
    if best_pts is not None and time.time() < deadline - 8:
        polished = _high_precision_polish(best_pts, deadline)
        update_best(polished)
    
    # Fallback to regular polygon if nothing worked
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normalize to unit area
    best_pts = _normalize_unit_area(best_pts)
    
    return best_pts
