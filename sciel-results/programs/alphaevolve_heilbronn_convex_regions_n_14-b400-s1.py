"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

def _min_triangle_area_vec(points):
    """Vectorized minimum triangle area for N points. Returns min area (not doubled)."""
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
    """Compute convex hull area. Returns 0 if degenerate."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Minimum triangle area divided by convex hull area."""
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _min_triangle_area_vec(points)
    return min_area / hull_area

def _all_triangle_areas(points):
    """Compute all triangle areas as a flat array."""
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
            tri_areas = 0.5 * np.abs(cross)
            if len(tri_areas) > 0:
                areas_list.append(tri_areas)
    if not areas_list:
        return np.array([0.0])
    return np.concatenate(areas_list)

def _objective_hard(points_flat):
    """Minimize hull_area / min_triangle_area."""
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
    """Soft-min objective for smoother optimization."""
    points = points_flat.reshape(N, 2)
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
    areas = _all_triangle_areas(points)
    if len(areas) == 0 or np.any(areas < 1e-12):
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min_val = -log_sum / beta
    hull_area = _hull_area_fast(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min_val

def _normalize_unit_area(points):
    """Scale points so convex hull area = 1."""
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

def _best_of_two(x1, x2):
    """Return the point set with better (lower) objective."""
    o1 = _objective_hard(x1)
    o2 = _objective_hard(x2)
    if o1 <= o2:
        return x1
    return x2

def _nm(x0, deadline, max_iter=3000, obj_func=None):
    """Run Nelder-Mead from x0."""
    if obj_func is None:
        obj_func = _objective_hard
    remaining = deadline - time.time()
    if remaining < 2.0:
        return None
    try:
        res = minimize(
            obj_func,
            x0,
            method='Nelder-Mead',
            options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
        )
        return res.x
    except Exception:
        return None

def _powell(x0, deadline, max_iter=1000, obj_func=None):
    """Run Powell from x0."""
    if obj_func is None:
        obj_func = _objective_hard
    remaining = deadline - time.time()
    if remaining < 3.0:
        return None
    try:
        res = minimize(
            obj_func,
            x0,
            method='Powell',
            options={'maxiter': max_iter, 'xtol': 1e-10, 'ftol': 1e-12}
        )
        return res.x
    except Exception:
        return None

def _lbfgs(x0, deadline, max_iter=500, obj_func=None):
    """Run L-BFGS-B from x0."""
    if obj_func is None:
        obj_func = _objective_soft
    remaining = deadline - time.time()
    if remaining < 3.0:
        return None
    try:
        res = minimize(
            obj_func,
            x0,
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        return res.x
    except Exception:
        return None

def _make_inits():
    """Generate diverse initializations."""
    inits = []
    rng = np.random.default_rng(42)
    
    # Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg_points = np.column_stack([np.cos(angles), np.sin(angles)])
    inits.append(reg_points.flatten())
    
    # Rotated regular polygons
    for rot in [0.1, 0.3, 0.7, 1.0, 1.5, 2.0]:
        a = angles + rot
        pts = np.column_stack([np.cos(a), np.sin(a)])
        inits.append(pts.flatten())
    
    # Jittered regular polygon
    for jitter_scale in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        jitter = rng.normal(0, jitter_scale, size=(N, 2))
        pts = reg_points + jitter
        inits.append(pts.flatten())
    
    # Random points on circle with varying radii
    for seed in range(8):
        rng_s = np.random.default_rng(seed * 1000 + 7)
        angles_r = np.sort(rng_s.uniform(0, 2 * np.pi, N))
        radii = rng_s.uniform(0.7, 1.3, N)
        pts = np.column_stack([radii * np.cos(angles_r), radii * np.sin(angles_r)])
        inits.append(pts.flatten())
    
    # Two concentric rings
    for r_inner in [0.3, 0.4, 0.5, 0.6, 0.7]:
        for n_inner in [5, 6, 7]:
            n_outer = N - n_inner
            a_inner = np.linspace(0, 2*np.pi, n_inner, endpoint=False)
            a_outer = np.linspace(0, 2*np.pi, n_outer, endpoint=False) + np.pi/n_outer
            pts_inner = np.column_stack([r_inner * np.cos(a_inner), r_inner * np.sin(a_inner)])
            pts_outer = np.column_stack([np.cos(a_outer), np.sin(a_outer)])
            pts = np.vstack([pts_inner, pts_outer])
            inits.append(pts.flatten())
    
    # Elliptical distributions
    for ax, ay in [(1.0, 0.5), (1.0, 0.7), (1.2, 0.8), (1.3, 0.9), (1.5, 0.7), (1.8, 0.6)]:
        angles_e = np.linspace(0, 2*np.pi, N, endpoint=False)
        pts = np.column_stack([ax * np.cos(angles_e), ay * np.sin(angles_e)])
        inits.append(pts.flatten())
    
    # Random convex-like: sort random points by angle from centroid
    for seed in range(8):
        rng_s = np.random.default_rng(seed * 777 + 13)
        pts_raw = rng_s.uniform(-1, 1, size=(N, 2))
        centroid = pts_raw.mean(axis=0)
        angles_c = np.arctan2(pts_raw[:,1] - centroid[1], pts_raw[:,0] - centroid[0])
        sorted_idx = np.argsort(angles_c)
        pts = pts_raw[sorted_idx]
        inits.append(pts.flatten())
    
    # Distorted square
    for distort in [0.0, 0.1, 0.2, 0.3, 0.4]:
        t = np.linspace(0, 1, 3, endpoint=False)
        bottom = np.column_stack([t + distort*(1-t)*t, np.zeros(3)])
        top = np.column_stack([1-t - distort*(1-t)*t, np.ones(3)])
        left = np.column_stack([np.zeros(3), t + distort*(1-t)*t])
        right = np.column_stack([np.ones(3), 1-t - distort*(1-t)*t])
        center = np.array([[0.5, 0.5], [0.5 + 0.1*distort, 0.5 - 0.1*distort]])
        pts = np.vstack([bottom, top, left, right, center])
        pts = pts - 0.5
        inits.append(pts.flatten())
    
    # Fibonacci spiral
    golden_angle = 2.399963229728653
    for r_scale in [1.0, 0.8, 1.2, 1.5]:
        pts_fib = np.zeros((N, 2))
        for i in range(N):
            angle = i * golden_angle
            r = r_scale * np.sqrt((i + 0.5) / N)
            pts_fib[i, 0] = r * np.cos(angle)
            pts_fib[i, 1] = r * np.sin(angle)
        inits.append(pts_fib.flatten())
    
    # Regular polygon with alternating radii
    for alt in [0.1, 0.2, 0.3, 0.4]:
        angles_alt = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii_alt = np.ones(N)
        radii_alt[::2] = 1.0 + alt
        radii_alt[1::2] = 1.0 - alt
        pts_alt = np.column_stack([radii_alt * np.cos(angles_alt), radii_alt * np.sin(angles_alt)])
        inits.append(pts_alt.flatten())
    
    # More random starts
    for seed in range(15):
        rng_s = np.random.default_rng(seed * 12345 + 999)
        pts_rand = rng_s.uniform(-1, 1, size=(N, 2))
        norms = np.linalg.norm(pts_rand, axis=1)
        mask = norms > 1
        pts_rand[mask] = pts_rand[mask] / norms[mask, None]
        angles_rand = np.arctan2(pts_rand[:,1], pts_rand[:,0])
        sorted_idx = np.argsort(angles_rand)
        pts_rand = pts_rand[sorted_idx]
        inits.append(pts_rand.flatten())
    
    # Hexagonal lattice-like
    for scale in [0.5, 0.7, 0.9]:
        pts_hex = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                x = scale * (i + 0.5 * (j % 2))
                y = scale * j * np.sqrt(3) / 2
                pts_hex.append([x, y])
        pts_hex = np.array(pts_hex[:N])
        inits.append(pts_hex.flatten())
    
    return inits

def construct():
    """
    Generate 14 points in a planar convex region normalized to unit area
    to maximize the minimum area over all triples.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    inits = _make_inits()
    max_inits = min(len(inits), 60)
    inits = inits[:max_inits]
    
    best_x = None
    best_obj = float('inf')
    
    # Phase 1: NM from each diverse start
    for idx, x0 in enumerate(inits):
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        
        time_per_init = min(6.0, max(2.0, remaining / max(1, len(inits) - idx)))
        
        res = _nm(x0, deadline, max_iter=int(time_per_init * 800))
        if res is not None:
            obj_val = _objective_hard(res)
            if obj_val < best_obj:
                best_obj = obj_val
                best_x = res.copy()
        
        # Also try soft-NM
        if time.time() < deadline - 8:
            res_soft = _nm(x0, deadline - 8, max_iter=400, obj_func=_objective_soft)
            if res_soft is not None:
                obj_val = _objective_hard(res_soft)
                if obj_val < best_obj:
                    best_obj = obj_val
                    best_x = res_soft.copy()
    
    # Phase 2: Powell polish
    if best_x is not None and time.time() < deadline - 10:
        res_powell = _powell(best_x, deadline, max_iter=2000)
        if res_powell is not None:
            obj_val = _objective_hard(res_powell)
            if obj_val < best_obj:
                best_obj = obj_val
                best_x = res_powell.copy()
    
    # Phase 3: Final NM polish
    if best_x is not None and time.time() < deadline - 5:
        res_final = _nm(best_x, deadline, max_iter=5000)
        if res_final is not None:
            obj_val = _objective_hard(res_final)
            if obj_val < best_obj:
                best_obj = obj_val
                best_x = res_final.copy()
    
    # Phase 4: Another Powell
    if best_x is not None and time.time() < deadline - 5:
        res_powell2 = _powell(best_x, deadline, max_iter=1000)
        if res_powell2 is not None:
            obj_val = _objective_hard(res_powell2)
            if obj_val < best_obj:
                best_obj = obj_val
                best_x = res_powell2.copy()
    
    # Phase 5: L-BFGS-B with soft objective
    if best_x is not None and time.time() < deadline - 5:
        res_lbfgs = _lbfgs(best_x, deadline, max_iter=500)
        if res_lbfgs is not None:
            obj_val = _objective_hard(res_lbfgs)
            if obj_val < best_obj:
                best_obj = obj_val
                best_x = res_lbfgs.copy()
    
    # Fallback
    if best_x is None:
        best_x = inits[0].copy()
    
    final_points = best_x.reshape(N, 2)
    final_points = _normalize_unit_area(final_points)
    
    # Safety check
    try:
        hull = ConvexHull(final_points)
        if hull.volume < 1e-10:
            angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
            reg_points = np.column_stack([np.cos(angles), np.sin(angles)])
            final_points = _normalize_unit_area(reg_points)
    except Exception:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        reg_points = np.column_stack([np.cos(angles), np.sin(angles)])
        final_points = _normalize_unit_area(reg_points)
    
    return final_points
# EVOLVE-BLOCK-END
