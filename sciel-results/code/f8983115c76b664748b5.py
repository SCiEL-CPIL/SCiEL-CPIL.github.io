"""
Hexagon packing: 11 unit hexagons inside a regular hexagon.
Strategy: enumerate polyhex topologies, then use multi-start simulated annealing
with per-hexagon independent rotation, followed by high-precision polish.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_rad):
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def compute_required_side_fast(inner, outer_center, outer_angle_deg):
    if not inner:
        return 0.0
    ox, oy = outer_center
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    max_proj = -np.inf
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        local = (verts - np.array([ox, oy])) @ R.T
        projs = local @ normals.T
        cur = projs.max()
        if cur > max_proj:
            max_proj = cur
    if max_proj < 0:
        max_proj = 0.0
    return 2.0 * max_proj / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=50.0):
    if not inner:
        return 0.0
    ox, oy = outer_center
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    all_projs = []
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        local = (verts - np.array([ox, oy])) @ R.T
        projs = local @ normals.T
        all_projs.append(projs)
    P = np.concatenate([p.ravel() for p in all_projs])
    if P.size == 0:
        return 0.0
    m = P.max()
    if m < -1e6:
        return 0.0
    smooth_max = m + math.log(np.sum(np.exp(beta * (P - m)))) / beta
    if smooth_max < 0:
        smooth_max = 0.0
    return 2.0 * smooth_max / SQRT3

def sat_overlap(v1, v2, eps=1e-9):
    axes = []
    for poly in (v1, v2):
        for i in range(len(poly)):
            p1 = poly[i]; p2 = poly[(i + 1) % len(poly)]
            e = (p2[0] - p1[0], p2[1] - p1[1])
            n = (-e[1], e[0])
            ln = math.hypot(n[0], n[1])
            if ln > 1e-12:
                axes.append((n[0] / ln, n[1] / ln))
    for ax in axes:
        min1 = min(p[0] * ax[0] + p[1] * ax[1] for p in v1)
        max1 = max(p[0] * ax[0] + p[1] * ax[1] for p in v1)
        min2 = min(p[0] * ax[0] + p[1] * ax[1] for p in v2)
        max2 = max(p[0] * ax[0] + p[1] * ax[1] for p in v2)
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    return True

def check_disjoint(inner, eps=1e-6):
    n = len(inner)
    polys = []
    for (x, y, a) in inner:
        polys.append(hex_vertices(x, y, 1.0, math.radians(a)))
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap(polys[i], polys[j], eps):
                return False
    return True

def min_pairwise_distance(inner):
    """Return the minimum distance between any pair of hexagon centers."""
    n = len(inner)
    min_d = float('inf')
    for i in range(n):
        xi, yi, _ = inner[i]
        for j in range(i + 1, n):
            xj, yj, _ = inner[j]
            d = math.hypot(xi - xj, yi - yj)
            if d < min_d:
                min_d = d
    return min_d

def overlap_penalty(inner):
    """Penalty based on minimum center distance (for same-orientation hexagons,
    the minimum center distance is sqrt(3) for non-overlapping)."""
    pen = 0.0
    n = len(inner)
    # For two unit hexagons with arbitrary rotation, the minimum center distance
    # for non-overlap depends on relative angle. A safe lower bound is sqrt(3)
    # (achieved when they share an edge with same orientation).
    # For different orientations, the required distance can be larger.
    # Use a conservative penalty.
    for i in range(n):
        xi, yi, ai = inner[i]
        for j in range(i + 1, n):
            xj, yj, aj = inner[j]
            d = math.hypot(xi - xj, yi - yj)
            # Required distance depends on relative angle
            rel_angle = abs(ai - aj) % 360
            if rel_angle > 180:
                rel_angle = 360 - rel_angle
            # For hexagons, the minimum distance for non-overlap is:
            # - same orientation (0 deg): sqrt(3)
            # - 30 deg: slightly more
            # - 60 deg: sqrt(3) again (by symmetry)
            # Conservative: use sqrt(3) + small margin
            min_req = SQRT3 + 0.001
            if d < min_req:
                pen += (min_req - d) ** 2
    return pen

def axial_to_xy(q, r, s=1.0):
    return (1.5 * s * q, SQRT3 * s * (r + q / 2.0))

def enumerate_polyhexes(n_cells):
    if n_cells == 1:
        return [frozenset({(0, 0)})]
    neighbors = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    prev = enumerate_polyhexes(n_cells - 1)
    seen = set()
    result = []
    for ph in prev:
        for (q, r) in ph:
            for dq, dr in neighbors:
                nq, nr = q + dq, r + dr
                if (nq, nr) in ph:
                    continue
                new_ph = frozenset(ph | {(nq, nr)})
                min_q = min(c[0] for c in new_ph)
                min_r = min(c[1] for c in new_ph)
                canon = frozenset((c[0] - min_q, c[1] - min_r) for c in new_ph)
                variants = set()
                for rot in range(6):
                    rotated = set()
                    for (cq, cr) in canon:
                        if rot == 0:
                            rotated.add((cq, cr))
                        elif rot == 1:
                            rotated.add((-cr, cq + cr))
                        elif rot == 2:
                            rotated.add((-cq - cr, cq))
                        elif rot == 3:
                            rotated.add((-cq, -cr))
                        elif rot == 4:
                            rotated.add((cr, -cq - cr))
                        elif rot == 5:
                            rotated.add((cq + cr, -cq))
                    mq = min(x[0] for x in rotated)
                    mr = min(x[1] for x in rotated)
                    variants.add(frozenset((x[0] - mq, x[1] - mr) for x in rotated))
                if variants not in seen:
                    seen.add(variants)
                    rep = min(variants, key=lambda v: sorted(v))
                    result.append(set(rep))
    return result

def build_lattice_config(axial_coords, s=1.0 + 1e-8):
    centers = [axial_to_xy(q, r, s) for q, r in axial_coords]
    return [[x, y, 0.0] for x, y in centers]

def simulated_annealing(inner_init, outer_center_init, outer_angle_init, deadline, seed=42):
    """
    Multi-parameter simulated annealing with per-hexagon independent rotation.
    Parameters: 3*N for inner (x, y, angle) + 3 for outer (cx, cy, angle)
    """
    np.random.seed(seed)
    
    def obj_smooth(params, beta=40.0):
        il = []
        for i in range(N):
            il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
        ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
        side = compute_required_side_smooth(il, (ocx, ocy), oa, beta=beta)
        pen = overlap_penalty(il)
        return side + 5000.0 * pen

    def obj_exact(params):
        il = []
        for i in range(N):
            il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
        ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
        side = compute_required_side_fast(il, (ocx, ocy), oa)
        pen = overlap_penalty(il)
        return side + 10000.0 * pen

    p0 = np.zeros(N * 3 + 3)
    for i, (x, y, a) in enumerate(inner_init):
        p0[i * 3] = x
        p0[i * 3 + 1] = y
        p0[i * 3 + 2] = a
    p0[N * 3] = outer_center_init[0]
    p0[N * 3 + 1] = outer_center_init[1]
    p0[N * 3 + 2] = outer_angle_init

    current = p0.copy()
    current_val = obj_smooth(current)
    best_params = current.copy()
    best_val = current_val

    T = 0.5
    T_min = 0.0005
    cool_rate = 0.9997
    max_iter = 8000

    for it in range(max_iter):
        if time.time() > deadline - 20:
            break
        nb = current.copy()
        r = np.random.random()
        scale = max(0.05, T)
        
        if r < 0.55:
            # Move a hexagon position
            idx = np.random.randint(0, N)
            dim = np.random.randint(0, 2)
            nb[idx * 3 + dim] += np.random.normal(0, 0.25 * scale)
        elif r < 0.75:
            # Rotate a hexagon
            idx = np.random.randint(0, N)
            nb[idx * 3 + 2] += np.random.normal(0, 15.0 * scale)
        elif r < 0.85:
            # Move outer center
            dim = np.random.randint(0, 2)
            nb[N * 3 + dim] += np.random.normal(0, 0.15 * scale)
        elif r < 0.95:
            # Rotate outer
            nb[N * 3 + 2] += np.random.normal(0, 5.0 * scale)
        else:
            # Move two hexagons (for better exploration)
            i1, i2 = np.random.choice(N, 2, replace=False)
            for idx in [i1, i2]:
                dim = np.random.randint(0, 2)
                nb[idx * 3 + dim] += np.random.normal(0, 0.15 * scale)

        nb_val = obj_smooth(nb)
        if nb_val < current_val or np.random.random() < math.exp(-(nb_val - current_val) / max(T, 1e-10)):
            current = nb
            current_val = nb_val
            if current_val < best_val:
                best_params = current.copy()
                best_val = current_val
        T *= cool_rate
        if T < T_min:
            break

    # Extract best
    best_inner = []
    for i in range(N):
        best_inner.append([float(best_params[i * 3]), float(best_params[i * 3 + 1]), float(best_params[i * 3 + 2])])
    best_oc = (float(best_params[N * 3]), float(best_params[N * 3 + 1]))
    best_oa = float(best_params[N * 3 + 2])
    return best_inner, best_oc, best_oa, best_params, best_val

def nelder_mead_polish(params, deadline):
    """High-precision Nelder-Mead polish with smooth then exact objective."""
    def obj_smooth(p, beta=60.0):
        il = []
        for i in range(N):
            il.append([p[i * 3], p[i * 3 + 1], p[i * 3 + 2]])
        ocx, ocy, oa = p[N * 3], p[N * 3 + 1], p[N * 3 + 2]
        side = compute_required_side_smooth(il, (ocx, ocy), oa, beta=beta)
        pen = overlap_penalty(il)
        return side + 5000.0 * pen

    def obj_exact(p):
        il = []
        for i in range(N):
            il.append([p[i * 3], p[i * 3 + 1], p[i * 3 + 2]])
        ocx, ocy, oa = p[N * 3], p[N * 3 + 1], p[N * 3 + 2]
        side = compute_required_side_fast(il, (ocx, ocy), oa)
        pen = overlap_penalty(il)
        return side + 20000.0 * pen

    current = params.copy()
    
    try:
        r1 = minimize(obj_smooth, current, method='Nelder-Mead',
                      options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 3000})
        current = r1.x
    except Exception:
        pass
    
    if time.time() < deadline - 10:
        try:
            r2 = minimize(obj_exact, current, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 3000})
            current = r2.x
        except Exception:
            pass
    
    # Coordinate descent fine polish
    if time.time() < deadline - 4:
        cur_val = obj_exact(current)
        step = 0.005
        for _ in range(60):
            if time.time() > deadline - 4:
                break
            improved = False
            for j in range(len(current)):
                for sign in [1, -1]:
                    trial = current.copy()
                    trial[j] += sign * step
                    trial_val = obj_exact(trial)
                    if trial_val < cur_val - 1e-13:
                        current = trial
                        cur_val = trial_val
                        improved = True
                        break
            if not improved:
                step *= 0.5
                if step < 1e-11:
                    break
    
    return current

def construct():
    t0 = time.time()
    deadline = t0 + 148.0  # 2 second safety margin

    # Enumerate polyhex topologies
    try:
        polyhexes = enumerate_polyhexes(N)
    except Exception:
        polyhexes = []
    
    if not polyhexes or time.time() > deadline - 100:
        polyhexes = []
    
    # Hand-crafted configs as fallback
    axial1 = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, -1), (1, 1), (-2, 1), (-1, -1)]
    axial2 = [(-1, -1), (0, -1), (1, -1), (2, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1), (2, 1)]
    axial3 = [(-1, -1), (0, -1), (1, -1), (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (-1, 1), (0, 1), (1, 1)]
    axial4 = [(-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1), (-1, 0), (0, 0), (1, 0), (-1, 1), (0, 1), (1, 1)]
    # Additional compact shapes
    axial5 = [(0, 0), (1, 0), (2, 0), (-1, 0), (-1, 1), (0, 1), (1, 1), (2, 1), (-1, -1), (0, -1), (1, -1)]
    axial6 = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1), (2, -1), (2, -2), (1, 1), (0, 2)]
    axial7 = [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1), (2, 1), (0, -1), (1, -1), (2, -1), (3, -1)]
    
    hand_configs = [set(axial1), set(axial2), set(axial3), set(axial4), set(axial5), set(axial6), set(axial7)]
    
    all_configs = list(polyhexes) + hand_configs
    seen_shapes = set()
    unique_configs = []
    for cfg in all_configs:
        min_q = min(c[0] for c in cfg)
        min_r = min(c[1] for c in cfg)
        canon = frozenset((c[0] - min_q, c[1] - min_r) for c in cfg)
        if canon not in seen_shapes:
            seen_shapes.add(canon)
            unique_configs.append(cfg)
    
    # Sort by compactness (bounding box area)
    def config_score(cfg):
        xs = [c[0] for c in cfg]
        ys = [c[1] for c in cfg]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))
    
    unique_configs.sort(key=config_score)
    max_configs = 15
    unique_configs = unique_configs[:max_configs]
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = None
    best_outer_angle = 0.0
    best_params = None
    
    # Phase 1: For each topology, run SA from multiple random starts
    for ci, axial in enumerate(unique_configs):
        if time.time() > deadline - 60:
            break
        
        s0 = 1.0 + 1e-8
        centers = [axial_to_xy(q, r, s0) for q, r in sorted(axial)]
        base_inner = [[x, y, 0.0] for x, y in centers]
        cx0 = sum(c[0] for c in centers) / len(centers)
        cy0 = sum(c[1] for c in centers) / len(centers)
        
        # Run SA with different seeds
        for seed in [42, 123, 777]:
            if time.time() > deadline - 55:
                break
            
            sa_inner, sa_oc, sa_oa, sa_params, sa_val = simulated_annealing(
                base_inner, (cx0, cy0), 0.0, deadline, seed=seed)
            
            if check_disjoint(sa_inner, eps=1e-5):
                sa_side = compute_required_side_fast(sa_inner, sa_oc, sa_oa)
                if sa_side < best_side:
                    best_side = sa_side
                    best_inner = sa_inner
                    best_outer_center = sa_oc
                    best_outer_angle = sa_oa
                    best_params = sa_params
    
    # Fallback if SA didn't find valid config
    if best_inner is None:
        axial = unique_configs[0] if unique_configs else axial1
        s0 = 1.0 + 1e-8
        centers = [axial_to_xy(q, r, s0) for q, r in axial]
        best_inner = [[x, y, 0.0] for x, y in centers]
        cx0 = sum(c[0] for c in best_inner) / len(best_inner)
        cy0 = sum(c[1] for c in best_inner) / len(best_inner)
        best_outer_center = (cx0, cy0)
        best_outer_angle = 0.0
        best_side = compute_required_side_fast(best_inner, best_outer_center, best_outer_angle)
        best_params = np.zeros(N * 3 + 3)
        for i, (x, y, a) in enumerate(best_inner):
            best_params[i * 3] = x
            best_params[i * 3 + 1] = y
            best_params[i * 3 + 2] = a
        best_params[N * 3] = cx0
        best_params[N * 3 + 1] = cy0
        best_params[N * 3 + 2] = 0.0
    
    # Phase 2: Nelder-Mead polish
    if time.time() < deadline - 15 and best_params is not None:
        polished = nelder_mead_polish(best_params, deadline)
        polish_inner = []
        for i in range(N):
            polish_inner.append([float(polished[i * 3]), float(polished[i * 3 + 1]), float(polished[i * 3 + 2])])
        polish_oc = (float(polished[N * 3]), float(polished[N * 3 + 1]))
        polish_oa = float(polished[N * 3 + 2])
        
        if check_disjoint(polish_inner, eps=1e-5):
            polish_side = compute_required_side_fast(polish_inner, polish_oc, polish_oa)
            if polish_side < best_side:
                best_inner = polish_inner
                best_outer_center = polish_oc
                best_outer_angle = polish_oa
                best_side = polish_side
    
    # Phase 3: Additional SA from the polished result
    if time.time() < deadline - 10:
        sa_inner, sa_oc, sa_oa, sa_params, sa_val = simulated_annealing(
            best_inner, best_outer_center, best_outer_angle, deadline, seed=999)
        
        if check_disjoint(sa_inner, eps=1e-5):
            sa_side = compute_required_side_fast(sa_inner, sa_oc, sa_oa)
            if sa_side < best_side:
                best_inner = sa_inner
                best_outer_center = sa_oc
                best_outer_angle = sa_oa
                best_side = sa_side
        
        # Quick polish of the new SA result
        if time.time() < deadline - 5 and sa_side < best_side + 0.01:
            p2 = np.zeros(N * 3 + 3)
            for i, (x, y, a) in enumerate(best_inner):
                p2[i * 3] = x
                p2[i * 3 + 1] = y
                p2[i * 3 + 2] = a
            p2[N * 3] = best_outer_center[0]
            p2[N * 3 + 1] = best_outer_center[1]
            p2[N * 3 + 2] = best_outer_angle
            polished2 = nelder_mead_polish(p2, deadline)
            p2_inner = []
            for i in range(N):
                p2_inner.append([float(polished2[i * 3]), float(polished2[i * 3 + 1]), float(polished2[i * 3 + 2])])
            p2_oc = (float(polished2[N * 3]), float(polished2[N * 3 + 1]))
            p2_oa = float(polished2[N * 3 + 2])
            
            if check_disjoint(p2_inner, eps=1e-5):
                p2_side = compute_required_side_fast(p2_inner, p2_oc, p2_oa)
                if p2_side < best_side:
                    best_inner = p2_inner
                    best_outer_center = p2_oc
                    best_outer_angle = p2_oa
                    best_side = p2_side
    
    # Final verification
    if not check_disjoint(best_inner, eps=1e-5):
        s0 = 1.0 + 1e-8
        centers = [axial_to_xy(q, r, s0) for q, r in axial1]
        best_inner = [[x, y, 0.0] for x, y in centers]
        cx0 = sum(c[0] for c in best_inner) / len(best_inner)
        cy0 = sum(c[1] for c in best_inner) / len(best_inner)
        best_outer_center = (cx0, cy0)
        best_outer_angle = 0.0
        best_side = compute_required_side_fast(best_inner, best_outer_center, best_outer_angle)
    
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
