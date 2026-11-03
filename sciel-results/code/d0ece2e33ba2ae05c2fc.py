"""
Heilbronn convex regions n=14: maximize min triangle area / convex hull area.

Strategy:
  1. Worst-triangle anchor: identify the bottleneck triangle, push its vertices
     outward while adjusting remaining points to maintain good coverage.
  2. Multi-beta soft objective for smooth gradient signal.
  3. Coordinate descent polish for precise local optimization.
  4. Multi-phase: diverse inits -> NM -> Powell -> L-BFGS -> coordinate descent.
  5. Time-guarded to finish within 150s.
"""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14

# Pre-compute all triangle index combinations
tri_indices = np.array(list(__import__('itertools').combinations(range(N), 3)))  # shape (364, 3)


def _min_triangle_area(points):
    """Minimum triangle area over all triples, vectorized."""
    n = points.shape[0]
    p = points
    pi = p[tri_indices[:, 0]]
    pj = p[tri_indices[:, 1]]
    pk = p[tri_indices[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    return np.min(areas)


def _min_triangle_area_with_indices(points):
    """Return (min_area, (i, j, k)) for the minimum-area triangle."""
    pi = points[tri_indices[:, 0]]
    pj = points[tri_indices[:, 1]]
    pk = points[tri_indices[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    idx = np.argmin(areas)
    return areas[idx], tuple(tri_indices[idx])


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


def _all_triangle_areas(points):
    pi = points[tri_indices[:, 0]]
    pj = points[tri_indices[:, 1]]
    pk = points[tri_indices[:, 2]]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return 0.5 * np.abs(cross)


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


def _objective_soft(flat, beta=50.0):
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


def _multi_beta_soft_obj(flat, betas=(20.0, 50.0, 100.0)):
    """Multi-beta soft objective for better gradient signal across scales."""
    pts = flat.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    areas = _all_triangle_areas(pts)
    if np.min(areas) < 1e-12:
        return 1e10
    total = 0.0
    for beta in betas:
        z = -beta * areas
        max_z = np.max(z)
        if max_z < -500:
            log_sum = max_z
        else:
            log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
        soft_min = -log_sum / beta
        if soft_min < 1e-12:
            return 1e10
        total += 1.0 / soft_min
    hull_area = _hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area * total / len(betas)


def _coordinate_descent_polish(pts, deadline, max_rounds=50, step=0.01):
    """
    Coordinate descent: for each point, try small perturbations in x and y
    to improve the minimum triangle area. This provides precise local optimization.
    """
    pts = pts.copy()
    best_score = _score(pts)
    
    for rnd in range(max_rounds):
        if time.time() > deadline - 5:
            break
        improved = False
        for i in range(N):
            # Try perturbations in x
            for dx in [-step, step]:
                pts_new = pts.copy()
                pts_new[i, 0] += dx
                s = _score(pts_new)
                if s > best_score + 1e-12:
                    pts = pts_new
                    best_score = s
                    improved = True
            # Try perturbations in y
            for dy in [-step, step]:
                pts_new = pts.copy()
                pts_new[i, 1] += dy
                s = _score(pts_new)
                if s > best_score + 1e-12:
                    pts = pts_new
                    best_score = s
                    improved = True
        # Reduce step size if no improvement
        if not improved:
            step *= 0.7
            if step < 1e-8:
                break
    
    return pts


def _worst_triangle_anchor(pts, deadline, max_iters=100):
    """
    Worst-triangle anchor strategy: identify the minimum-area triangle,
    push its vertices outward to increase its area while adjusting other points.
    """
    pts = pts.copy()
    best_score = _score(pts)
    
    for it in range(max_iters):
        if time.time() > deadline - 10:
            break
        
        # Find the worst (minimum area) triangle
        min_area, (i, j, k) = _min_triangle_area_with_indices(pts)
        
        if min_area < 1e-10:
            break
        
        # Compute centroid of the worst triangle
        centroid = np.mean(pts[[i, j, k]], axis=0)
        
        # Compute hull center
        try:
            hull = ConvexHull(pts)
            hull_center = np.mean(pts[hull.vertices], axis=0)
        except Exception:
            hull_center = np.mean(pts, axis=0)
        
        # Direction from hull center to each vertex of the worst triangle
        # Push vertices outward along direction from hull center
        for idx in [i, j, k]:
            direction = pts[idx] - hull_center
            norm = np.linalg.norm(direction)
            if norm < 1e-10:
                direction = np.array([1.0, 0.0])
                norm = 1.0
            else:
                direction = direction / norm
            
            # Try pushing outward
            for push in [0.05, 0.02, 0.01]:
                pts_new = pts.copy()
                pts_new[idx] = pts[idx] + push * direction
                s = _score(pts_new)
                if s > best_score + 1e-10:
                    pts = pts_new
                    best_score = s
                    break
        
        # Also try pulling other points slightly toward center to maintain balance
        for idx in range(N):
            if idx not in [i, j, k]:
                direction = hull_center - pts[idx]
                norm = np.linalg.norm(direction)
                if norm < 1e-10:
                    continue
                direction = direction / norm
                for pull in [0.01, 0.005]:
                    pts_new = pts.copy()
                    pts_new[idx] = pts[idx] + pull * direction
                    s = _score(pts_new)
                    if s > best_score + 1e-10:
                        pts = pts_new
                        best_score = s
                        break
        
        # Check for stagnation
        current_score = _score(pts)
        if abs(current_score - best_score) < 1e-12:
            break
    
    return pts


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


# ---------- Initial configurations ----------

def _initial_regular_polygon():
    """Regular 14-gon."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return np.column_stack([np.cos(angles), np.sin(angles)])


def _initial_mixed():
    """Two concentric rings (7+7) with rotation."""
    inits = []
    for r1 in [0.3, 0.5, 0.7, 0.85]:
        a1 = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        a2 = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
        pts = np.vstack([
            r1 * np.column_stack([np.cos(a1), np.sin(a1)]),
            np.column_stack([np.cos(a2), np.sin(a2)])
        ])
        inits.append(pts)
    return inits


def _initial_hexagonal():
    """Hexagonal lattice-inspired configurations."""
    inits = []
    rng = np.random.default_rng(42)
    for seed in range(5):
        rs = np.random.default_rng(seed * 13 + 7)
        # Start with hexagonal grid and perturb
        pts = []
        # 3 rows: 5, 4, 5 points
        for row, (y, n_pts) in enumerate([(0, 5), (0.5, 4), (1.0, 5)]):
            for col in range(n_pts):
                x = col * 0.8 + (0.4 if row % 2 else 0)
                pts.append([x + rs.normal(0, 0.1), y + rs.normal(0, 0.1)])
        pts = np.array(pts[:N])
        # Normalize to unit area
        pts = _normalize_unit_area(pts)
        inits.append(pts)
    return inits


def _initial_cosine_mod():
    """Cosine-modulated radius configurations."""
    inits = []
    for k in [1, 2, 3]:
        for amp in [0.1, 0.2, 0.3]:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            radii = 1.0 + amp * np.cos(k * angles)
            pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            inits.append(pts)
    return inits


def _make_initial_points():
    """Generate diverse initial point configurations."""
    inits = []
    
    # Regular polygon
    inits.append(_initial_regular_polygon())
    
    # Mixed ring configurations
    inits.extend(_initial_mixed())
    
    # Hexagonal
    inits.extend(_initial_hexagonal())
    
    # Cosine-modulated
    inits.extend(_initial_cosine_mod())
    
    # Random configurations
    for seed in range(10):
        rs = np.random.default_rng(seed * 17 + 3)
        angles = np.sort(rs.uniform(0, 2 * np.pi, N))
        radii = rs.uniform(0.4, 1.2, N)
        pts = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        inits.append(pts)
    
    # Jittered regular polygon
    base = _initial_regular_polygon()
    rng = np.random.default_rng(42)
    for js in [0.05, 0.1, 0.2]:
        inits.append(base + rng.normal(0, js, size=(N, 2)))
    
    # Elliptical configurations
    for rx, ry in [(1.0, 0.6), (1.0, 0.8), (1.2, 0.7)]:
        a = np.linspace(0, 2 * np.pi, N, endpoint=False)
        inits.append(np.column_stack([rx * np.cos(a), ry * np.sin(a)]))
    
    return inits


# ---------- Optimizer helpers ----------

def _nm(flat0, deadline, obj, max_iter=2000):
    if time.time() > deadline - 3:
        return None
    try:
        res = minimize(obj, flat0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
        return res
    except Exception:
        return None


def _powell(flat0, deadline, obj, max_iter=1000):
    if time.time() > deadline - 3:
        return None
    try:
        res = minimize(obj, flat0, method='Powell',
                       options={'maxiter': max_iter, 'xtol': 1e-11, 'ftol': 1e-13})
        return res
    except Exception:
        return None


def _lbfgs(flat0, deadline, obj, max_iter=3000):
    if time.time() > deadline - 5:
        return None
    try:
        res = minimize(obj, flat0, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
        return res
    except Exception:
        return None


# ---------- Main ----------

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill

    best_score = 0.0
    best_pts = None

    def _consider(pts):
        nonlocal best_score, best_pts
        pts = np.asarray(pts, dtype=float)
        if pts.shape != (N, 2) or np.any(~np.isfinite(pts)):
            return
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()

    # Phase 1: Nelder-Mead on diverse initializations with soft objective
    cart_inits = _make_initial_points()
    for i, pts0 in enumerate(cart_inits):
        if time.time() > deadline - 80:
            break
        res = _nm(pts0.flatten(), deadline, _objective_soft, max_iter=500)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Phase 2: Powell refinement from best
    if best_pts is not None and time.time() < deadline - 60:
        res = _powell(best_pts.flatten(), deadline, _objective_soft, max_iter=500)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Phase 3: L-BFGS on multi-beta soft objective
    if best_pts is not None and time.time() < deadline - 50:
        res = _lbfgs(best_pts.flatten(), deadline, _multi_beta_soft_obj, max_iter=2000)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Phase 4: Worst-triangle anchor strategy
    if best_pts is not None and time.time() < deadline - 35:
        anchored = _worst_triangle_anchor(best_pts, deadline, max_iters=50)
        _consider(anchored)

    # Phase 5: Coordinate descent polish
    if best_pts is not None and time.time() < deadline - 25:
        polished = _coordinate_descent_polish(best_pts, deadline, max_rounds=30, step=0.02)
        _consider(polished)

    # Phase 6: Hard objective NM from best
    if best_pts is not None and time.time() < deadline - 20:
        res = _nm(best_pts.flatten(), deadline, _objective_hard, max_iter=800)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Phase 7: Differential evolution (global search) if time permits
    if time.time() < deadline - 30:
        try:
            bounds = [(-1.5, 1.5)] * (N * 2)
            res_de = differential_evolution(
                _objective_soft, bounds,
                seed=42, maxiter=20, tol=1e-10,
                popsize=10, mutation=(0.3, 1.0), recombination=0.7,
                polish=False, update_interval=1
            )
            if res_de is not None and np.isfinite(res_de.fun) and res_de.fun < 1e9:
                _consider(res_de.x.reshape(N, 2))
        except Exception:
            pass

    # Phase 8: Local refinement after DE
    if best_pts is not None and time.time() < deadline - 15:
        res = _nm(best_pts.flatten(), deadline, _objective_hard, max_iter=1000)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Phase 9: Final coordinate descent polish
    if best_pts is not None and time.time() < deadline - 10:
        polished = _coordinate_descent_polish(best_pts, deadline, max_rounds=50, step=0.01)
        _consider(polished)

    # Phase 10: Final L-BFGS polish
    if best_pts is not None and time.time() < deadline - 5:
        res = _lbfgs(best_pts.flatten(), deadline, _multi_beta_soft_obj, max_iter=1000)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Fallback
    if best_pts is None:
        best_pts = _initial_regular_polygon()

    final_points = _normalize_unit_area(best_pts)

    # Safety check
    try:
        hull = ConvexHull(final_points)
        if hull.volume < 1e-10:
            final_points = _normalize_unit_area(_initial_regular_polygon())
    except Exception:
        final_points = _normalize_unit_area(_initial_regular_polygon())

    return final_points
