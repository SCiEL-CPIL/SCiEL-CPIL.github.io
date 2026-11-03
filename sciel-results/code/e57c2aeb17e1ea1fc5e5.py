"""
Heilbronn convex regions n=14: maximize min triangle area / convex hull area.

Strategy: Centrally symmetric parameterization (7 antipodal pairs) with
smooth, low-frequency Fourier shape deformations. The 7 independent vertices
are placed at angles θ_i = base_i + δ_i with radii r_i, and the full 14-point
set is {v_1..v_7, -v_1..-v_7}. This exploits the known symmetry of optimal
Heilbronn configurations for even N, halving the effective search space while
guaranteeing a valid convex polygon (when angles are sorted and radii positive).

Optimization: Multi-phase with L-BFGS-B (smooth), Nelder-Mead, Powell, and
Differential Evolution in the 14D parameter space (7 angles + 7 radii),
followed by Cartesian refinement.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import itertools

N = 14
HALF = 7

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


def _decode_sym(params):
    """
    Decode 14 params [theta_1..theta_7, r_1..r_7] to 14 centrally symmetric points.
    The 7 independent vertices are at angles theta_i with radii r_i.
    The other 7 are their negatives.
    Points are sorted by angle to ensure convex polygon ordering.
    """
    thetas = params[:HALF]
    radii = np.maximum(params[HALF:], 1e-8)
    
    half_x = radii * np.cos(thetas)
    half_y = radii * np.sin(thetas)
    
    half = np.column_stack([half_x, half_y])
    full = np.vstack([half, -half])
    
    # Sort by angle around origin for convex ordering
    angles = np.arctan2(full[:, 1], full[:, 0])
    order = np.argsort(angles)
    full = full[order]
    
    return full


def _objective_sym_hard(params):
    if np.any(~np.isfinite(params)):
        return 1e10
    pts = _decode_sym(params)
    if np.any(~np.isfinite(pts)):
        return 1e10
    min_area = _min_triangle_area(pts)
    if min_area < 1e-12:
        return 1e10
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / min_area


def _objective_sym_soft(params, beta=100.0):
    if np.any(~np.isfinite(params)):
        return 1e10
    pts = _decode_sym(params)
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


def _objective_cart_hard(flat):
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


def _objective_cart_soft(flat, beta=80.0):
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


def _generate_sym_inits():
    """Generate diverse initial parameter vectors for centrally symmetric optimization."""
    inits = []
    base_angles = np.linspace(0, 2 * np.pi, HALF, endpoint=False)
    
    # 1. Regular heptagon
    inits.append(np.concatenate([base_angles, np.ones(HALF)]))
    
    # 2. Rotated regular heptagons
    for rot in [0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0]:
        a = base_angles + rot
        inits.append(np.concatenate([a, np.ones(HALF)]))
    
    # 3. Ellipse-like (varying radii with 2-fold pattern)
    for rx, ry in [(1.2, 0.8), (0.8, 1.2), (1.3, 0.7), (0.7, 1.3), (1.1, 0.9), (1.15, 0.85)]:
        a = base_angles
        r = np.ones(HALF)
        # Approximate ellipse: r(θ) ≈ 1 / sqrt(cos²/rx² + sin²/ry²)
        for i in range(HALF):
            c = np.cos(a[i])
            s = np.sin(a[i])
            r[i] = 1.0 / np.sqrt((c/rx)**2 + (s/ry)**2)
        inits.append(np.concatenate([a, r]))
    
    # 4. Perturbed regular
    rng = np.random.default_rng(42)
    for js in [0.03, 0.08, 0.15, 0.25]:
        a = base_angles + rng.normal(0, js, HALF)
        r = 1.0 + rng.normal(0, js * 0.5, HALF)
        r = np.maximum(r, 0.5)
        inits.append(np.concatenate([a, r]))
    
    # 5. Random smooth shapes
    for seed in range(30):
        rs = np.random.default_rng(seed * 31 + 17)
        a = base_angles + rs.normal(0, 0.2, HALF)
        a = np.sort(a)
        r = 1.0 + rs.normal(0, 0.15, HALF)
        r = np.maximum(r, 0.5)
        inits.append(np.concatenate([a, r]))
    
    # 6. Shapes with low-frequency modulation
    for k_dom in [1, 2, 3]:
        for amp in [0.05, 0.1, 0.2, 0.3]:
            a = base_angles
            r = np.ones(HALF)
            for i in range(HALF):
                r[i] = 1.0 + amp * np.cos(k_dom * a[i])
            r = np.maximum(r, 0.5)
            inits.append(np.concatenate([a, r]))
            # With sine
            r2 = np.ones(HALF)
            for i in range(HALF):
                r2[i] = 1.0 + amp * np.sin(k_dom * a[i])
            r2 = np.maximum(r2, 0.5)
            inits.append(np.concatenate([a, r2]))
    
    # 7. Non-uniform angles
    for seed in range(15):
        rs = np.random.default_rng(seed * 43 + 29)
        a = np.sort(rs.uniform(0, 2 * np.pi, HALF))
        r = 1.0 + rs.normal(0, 0.1, HALF)
        r = np.maximum(r, 0.5)
        inits.append(np.concatenate([a, r]))
    
    return inits


def _nm(x0, deadline, obj, max_iter=2000):
    if time.time() > deadline - 3:
        return None
    try:
        res = minimize(obj, x0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True})
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
                       options={'maxiter': max_iter, 'ftol': 1e-15, 'gtol': 1e-11, 'maxls': 50})
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


def _hull_pull(points, alpha=0.15):
    """Expand convex hull vertices outward to enlarge hull area."""
    try:
        hull = ConvexHull(points)
        hull_idx = set(hull.vertices.tolist())
    except Exception:
        return points.copy()
    centroid = points.mean(axis=0)
    new_pts = points.copy()
    for idx in hull_idx:
        v = points[idx] - centroid
        norm = np.linalg.norm(v)
        if norm > 1e-12:
            new_pts[idx] = points[idx] + alpha * v
    return new_pts


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # 15s safety margin before 150s kill
    
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
            best_obj = _objective_cart_hard(pts.flatten())
    
    # Phase 1: L-BFGS-B on symmetric parameters (14D smooth space)
    sym_inits = _generate_sym_inits()
    
    for i, p0 in enumerate(sym_inits):
        if time.time() > deadline - 65:
            break
        res = _lbfgs(p0, deadline, _objective_sym_soft, max_iter=2000)
        if res is not None:
            pts = _decode_sym(res.x)
            update_best(pts)
    
    # Phase 2: NM refinement on symmetric parameters
    for i, p0 in enumerate(sym_inits[:20]):
        if time.time() > deadline - 50:
            break
        res = _nm(p0, deadline, _objective_sym_hard, max_iter=1500)
        if res is not None and res.fun < best_obj:
            pts = _decode_sym(res.x)
            update_best(pts)
    
    # Phase 3: Cartesian local refinement from best found
    if best_pts is not None and time.time() < deadline - 35:
        res = _lbfgs(best_pts.flatten(), deadline, _objective_cart_soft, max_iter=3000)
        if res is not None and res.fun < best_obj:
            update_best(res.x.reshape(N, 2))
        
        # Try hull pull
        if time.time() < deadline - 25:
            pulled = _hull_pull(best_pts, alpha=0.15)
            update_best(pulled)
            if time.time() < deadline - 20:
                res = _lbfgs(pulled.flatten(), deadline, _objective_cart_soft, max_iter=2000)
                if res is not None and res.fun < best_obj:
                    update_best(res.x.reshape(N, 2))
    
    # Phase 4: NM on Cartesian
    if best_pts is not None and time.time() < deadline - 20:
        res = _nm(best_pts.flatten(), deadline, _objective_cart_hard, max_iter=2000)
        if res is not None and res.fun < best_obj:
            update_best(res.x.reshape(N, 2))
    
    # Phase 5: Differential Evolution in symmetric parameter space
    if time.time() < deadline - 20:
        try:
            bounds = [(-np.pi, np.pi)] * HALF + [(0.3, 3.0)] * HALF
            de_res = differential_evolution(
                _objective_sym_hard, bounds,
                seed=42, maxiter=30, tol=1e-8,
                popsize=12, mutation=(0.5, 1.0), recombination=0.7,
                polish=False, update_interval=1,
                init='sobol'
            )
            if de_res.fun < best_obj:
                pts = _decode_sym(de_res.x)
                update_best(pts)
        except Exception:
            pass
    
    # Phase 6: Final polish with multiple methods
    if best_pts is not None and time.time() < deadline - 5:
        res = _lbfgs(best_pts.flatten(), deadline, _objective_cart_soft, max_iter=3000)
        if res is not None and res.fun < best_obj:
            update_best(res.x.reshape(N, 2))
        
        if time.time() < deadline - 3:
            res = _powell(best_pts.flatten(), deadline, _objective_cart_hard, max_iter=1000)
            if res is not None and res.fun < best_obj:
                update_best(res.x.reshape(N, 2))
    
    # Fallback to regular polygon if nothing worked
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Normalize to unit area
    best_pts = _normalize_unit_area(best_pts)
    
    return best_pts
