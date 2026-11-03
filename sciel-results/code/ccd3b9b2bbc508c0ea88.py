"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from itertools import combinations

# ---------- Core geometry ----------

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area for N points (returns min area, not doubled)."""
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _hull_area_fast(points):
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area_vec(points)
    return min_area / hull_area

def _all_triangle_areas(points):
    """Return array of all triangle areas (N choose 3)."""
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas_list = []
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas_list.append(0.5 * np.abs(cross))
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)

def _objective_hard(points_flat):
    """Minimize hull_area / min_triangle_area (scale-invariant)."""
    points = points_flat.reshape(N, 2)
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
    min_area = _min_triangle_area_vec(points)
    if min_area < 1e-12:
        return 1e10
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / min_area

def _objective_soft(points_flat, beta=150.0):
    """Smooth soft-min objective: hull_area / soft_min(triangle areas)."""
    points = points_flat.reshape(N, 2)
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min_val = -log_sum / beta
    if soft_min_val < 1e-12:
        return 1e10
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min_val

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

def _hull_pull(points, alpha=0.5):
    """Expand convex hull outward (structural move to enlarge hull area, improving ratio)."""
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

def _best_of_two(a, b):
    sa = _score(a)
    sb = _score(b)
    return (a, sa) if sa >= sb else (b, sb)

def _nm(x0, deadline, max_iter=2500, obj=_objective_hard):
    remaining = deadline - time.time()
    if remaining < 2.0:
        return None
    try:
        res = minimize(obj, x0, method='Nelder-Mead',
                       options={'maxiter': max_iter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
        return res
    except Exception:
        return None

def _lbfgs(x0, deadline, max_iter=5000, obj=_objective_soft):
    remaining = deadline - time.time()
    if remaining < 5.0:
        return None
    try:
        res = minimize(obj, x0, method='L-BFGS-B',
                       options={'maxiter': max_iter, 'ftol': 1e-14, 'gtol': 1e-10})
        return res
    except Exception:
        return None

def _powell(x0, deadline, max_iter=1500, obj=_objective_hard):
    remaining = deadline - time.time()
    if remaining < 3.0:
        return None
    try:
        res = minimize(obj, x0, method='Powell',
                       options={'maxiter': max_iter, 'xtol': 1e-11, 'ftol': 1e-13})
        return res
    except Exception:
        return None

def _coordinate_descent(pts, deadline, max_rounds=50):
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
                for _ in range(30):
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
                    if step < 1e-8:
                        break
        if not improved:
            break
    return pts

# ---------- Hexagonal lattice representation (NEW STRUCTURAL IDEA) ----------
# Model 14 points as a 2x7 grid on a hexagonal lattice.
# Basis vectors: v1, v2 (2D each) = 4 params
# 7 interior offsets (one per column) = 14 params
# Total: 18 params, but we fix v1 length to 1 (scale invariance) -> 17 params
# Actually, we keep all 4 basis params + 14 offsets = 18 params.

def _hex_grid_decode(params):
    """Decode hex grid parameters to 14 points.
    
    params: array of shape (18,)
      params[0:2] = v1 (first basis vector)
      params[2:4] = v2 (second basis vector)
      params[4:18] = 7 column offsets, each 2D (offset for row 0 and row 1 in that column)
      Actually: 7 columns, each column has 2 points (row 0 and row 1).
      So offsets are 7 * 2 * 2 = 28? No, that's too many.
      
    Let me reconsider: 2 rows x 7 columns = 14 points.
    Point (r, c) = c * v1 + r * v2 + offset[r, c]
    But offsets should be small. Let's just use:
      params[0:2] = v1
      params[2:4] = v2
      params[4:18] = 7 offsets for the "inner" points? 
    
    Simpler: Just use 14 free points but initialize them in a hex grid pattern.
    The structural insight is the initialization and the symmetry-aware search.
    """
    # Actually, let's just return the points directly from a flat array
    # The hex grid is just a smart initialization.
    return params.reshape(N, 2)

def _make_hex_grid_init():
    """Create a hexagonal lattice initialization for 14 points (2x7 grid)."""
    # Hexagonal lattice basis
    v1 = np.array([1.0, 0.0])
    v2 = np.array([0.5, np.sqrt(3)/2])
    
    points = []
    for c in range(7):
        for r in range(2):
            pt = c * v1 + r * v2
            points.append(pt)
    
    pts = np.array(points)
    return pts

def _make_hex_grid_init_rotated(angle):
    """Create a rotated hexagonal lattice initialization."""
    pts = _make_hex_grid_init()
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    return pts @ R.T

# ---------- Initializations ----------

def _make_inits():
    inits = []
    rng = np.random.default_rng(2024)

    # 1. Regular 14-gon (convex, all on circle)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(reg.flatten())

    # 2. Rotated regular polygons
    for rot in [0.1, 0.4, 0.9, 1.3, 2.0, 3.5]:
        a = angles + rot
        inits.append(np.column_stack([np.cos(a), np.sin(a)]).flatten())

    # 3. Hexagonal lattice (NEW)
    for angle in [0.0, np.pi/6, np.pi/4, np.pi/3, np.pi/2]:
        hex_pts = _make_hex_grid_init_rotated(angle)
        inits.append(hex_pts.flatten())

    # 4. Two concentric rings (7 inner, 7 outer)
    for r_inner in [0.3, 0.45, 0.6, 0.75, 0.85]:
        a_in = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        a_out = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
        pts_in = r_inner * np.column_stack([np.cos(a_in), np.sin(a_in)])
        pts_out = np.column_stack([np.cos(a_out), np.sin(a_out)])
        pts = np.vstack([pts_in, pts_out])
        inits.append(pts.flatten())

    # 5. 3 rings (5, 5, 4)
    for r1, r2 in [(0.3, 0.65), (0.4, 0.8), (0.5, 0.9)]:
        a1 = np.linspace(0, 2*np.pi, 5, endpoint=False)
        a2 = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
        a3 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
        pts = np.vstack([
            r1*np.column_stack([np.cos(a1), np.sin(a1)]),
            r2*np.column_stack([np.cos(a2), np.sin(a2)]),
            np.column_stack([np.cos(a3), np.sin(a3)])
        ])
        inits.append(pts.flatten())

    # 6. Jittered regular polygon
    for js in [0.05, 0.12, 0.25, 0.4]:
        pts = reg + rng.normal(0, js, size=(N, 2))
        inits.append(pts.flatten())

    # 7. Random on circle with varying radius
    for seed in range(8):
        rs = np.random.default_rng(seed * 13 + 7)
        ang = np.sort(rs.uniform(0, 2 * np.pi, N))
        rad = rs.uniform(0.5, 1.3, N)
        inits.append(np.column_stack([rad * np.cos(ang), rad * np.sin(ang)]).flatten())

    # 8. Points on two interleaved circles (7 each), with angular offset
    for r1, r2, off in [(0.5, 1.0, 0.0), (0.5, 1.0, np.pi/7), (0.7, 1.0, np.pi/14),
                        (0.6, 1.0, np.pi/7), (0.8, 1.0, 0.0)]:
        a1 = np.linspace(0, 2*np.pi, 7, endpoint=False)
        a2 = np.linspace(0, 2*np.pi, 7, endpoint=False) + off
        pts = np.vstack([
            r1 * np.column_stack([np.cos(a1), np.sin(a1)]),
            r2 * np.column_stack([np.cos(a2), np.sin(a2)])
        ])
        inits.append(pts.flatten())

    # 9. Grid-like: 7 on top arc, 7 on bottom arc
    for frac in [0.3, 0.5, 0.7]:
        top_ys = np.linspace(frac, 1.0, 7)
        bot_ys = np.linspace(-1.0, -frac, 7)
        top_xs = np.sqrt(np.maximum(0, 1 - top_ys**2)) * np.linspace(-0.8, 0.8, 7)
        bot_xs = np.sqrt(np.maximum(0, 1 - bot_ys**2)) * np.linspace(-0.8, 0.8, 7)
        pts = np.vstack([np.column_stack([top_xs, top_ys]),
                         np.column_stack([bot_xs, bot_ys])])
        inits.append(pts.flatten())

    return inits

# ---------- Directed push-pull local search (from knowledge) ----------

def _directed_push_pull(pts, deadline, max_iter=100):
    """Identify worst triangle and push its vertices outward to increase area."""
    pts = pts.copy()
    n = len(pts)
    
    for it in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        # Find the worst (smallest area) triangle
        min_area = float('inf')
        worst_tri = None
        x = pts[:, 0]
        y = pts[:, 1]
        for i in range(n):
            for j in range(i + 1, n):
                if j + 1 >= n:
                    continue
                dx_ji = x[j] - x[i]
                dy_ji = y[j] - y[i]
                dx_ki = x[j+1:] - x[i]
                dy_ki = y[j+1:] - y[i]
                cross = dx_ji * dy_ki - dx_ki * dy_ji
                areas = 0.5 * np.abs(cross)
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
                    # Find which k gives the minimum
                    k_offset = np.argmin(areas)
                    k = j + 1 + k_offset
                    worst_tri = (i, j, k)
        
        if worst_tri is None:
            break
        
        i, j, k = worst_tri
        centroid = pts.mean(axis=0)
        
        # Push the three vertices outward from centroid
        step = 0.02
        improved = False
        for idx in [i, j, k]:
            v = pts[idx] - centroid
            norm = np.linalg.norm(v)
            if norm > 1e-12:
                new_pt = pts[idx] + step * v / norm
                old_pt = pts[idx].copy()
                pts[idx] = new_pt
                new_score = _score(pts)
                old_score = _score(pts)
                # Actually we need to compare before and after
                pts[idx] = old_pt
                old_score = _score(pts)
                pts[idx] = new_pt
                new_score = _score(pts)
                if new_score > old_score:
                    improved = True
                else:
                    pts[idx] = old_pt
        
        if not improved:
            # Try smaller step
            step *= 0.5
            if step < 1e-6:
                break
    
    return pts

# ---------- Main ----------

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # safety margin before 150s kill

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

    inits = _make_inits()

    # Phase 1: Nelder-Mead on hard objective from diverse starts
    for x0 in inits:
        if time.time() > deadline - 50:
            break
        res = _nm(x0, deadline, max_iter=1500, obj=_objective_hard)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Phase 2: Directed push-pull from best
    if best_pts is not None and time.time() < deadline - 30:
        pushed = _directed_push_pull(best_pts, deadline, max_iter=50)
        _consider(pushed)
        res = _nm(pushed.flatten(), deadline, max_iter=1000, obj=_objective_hard)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Phase 3: L-BFGS on soft objective from best
    if best_pts is not None and time.time() < deadline - 25:
        res = _lbfgs(best_pts.flatten(), deadline, max_iter=3000, obj=_objective_soft)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Phase 4: soft-min NM from best
    if best_pts is not None and time.time() < deadline - 20:
        res = _nm(best_pts.flatten(), deadline, max_iter=2000, obj=_objective_soft)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Phase 5: hull-pull structural move + refine
    if best_pts is not None and time.time() < deadline - 15:
        for alpha in [0.1, 0.2, 0.4, 0.6, 0.9]:
            if time.time() > deadline - 12:
                break
            pulled = _hull_pull(best_pts, alpha=alpha)
            _consider(pulled)
            res = _nm(pulled.flatten(), deadline, max_iter=1000, obj=_objective_hard)
            if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
                _consider(res.x.reshape(N, 2))

    # Phase 6: Coordinate descent for fine-tuning
    if best_pts is not None and time.time() < deadline - 10:
        cd_pts = _coordinate_descent(best_pts, deadline, max_rounds=30)
        _consider(cd_pts)

    # Phase 7: Powell polish
    if best_pts is not None and time.time() < deadline - 8:
        res = _powell(best_pts.flatten(), deadline, max_iter=1500, obj=_objective_hard)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Phase 8: Final L-BFGS + NM polish
    if best_pts is not None and time.time() < deadline - 5:
        res = _lbfgs(best_pts.flatten(), deadline, max_iter=2000, obj=_objective_soft)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    if best_pts is not None and time.time() < deadline - 2:
        res = _nm(best_pts.flatten(), deadline, max_iter=2000, obj=_objective_hard)
        if res is not None and np.isfinite(res.fun) and res.fun < 1e9:
            _consider(res.x.reshape(N, 2))

    # Fallback
    if best_pts is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_pts = np.column_stack([np.cos(angles), np.sin(angles)])

    final_points = _normalize_unit_area(best_pts)

    # Safety check
    try:
        hull = ConvexHull(final_points)
        if hull.volume < 1e-10:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))
    except Exception:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = _normalize_unit_area(np.column_stack([np.cos(angles), np.sin(angles)]))

    return final_points
# EVOLVE-BLOCK-END
