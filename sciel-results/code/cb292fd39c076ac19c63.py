"""
Hexagon packing: 11 unit hexagons in minimal outer hexagon.
Rigid-body 1D global orientation optimization over polyhex topologies,
plus local relaxation and high-precision polish.
"""
import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

_UNIT_VERTS = np.column_stack([np.cos(np.arange(6) * np.pi / 3.0),
                               np.sin(np.arange(6) * np.pi / 3.0)])
_NORMALS = np.array([[math.cos(math.radians(30 + i * 60)),
                      math.sin(math.radians(30 + i * 60))] for i in range(6)])
_UNIT_VERTS_T = _UNIT_VERTS.T  # shape (6,2)

def hex_vertices(cx, cy, s, angle_rad):
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return verts

def _get_rotated_verts(angle_rad):
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return _UNIT_VERTS @ R.T

def compute_required_side_fast(inner, outer_center, outer_angle_deg):
    if not inner:
        return 0.0
    ox, oy = outer_center
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    max_proj = -np.inf
    for (ix, iy, iang) in inner:
        verts = _get_rotated_verts(math.radians(iang))
        local = (verts + np.array([ix - ox, iy - oy])) @ R.T
        projs = local @ _NORMALS.T
        cur = projs.max()
        if cur > max_proj:
            max_proj = cur
    if max_proj < 0:
        max_proj = 0.0
    return 2.0 * max_proj / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=60.0):
    if not inner:
        return 0.0
    ox, oy = outer_center
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    all_projs = []
    for (ix, iy, iang) in inner:
        verts = _get_rotated_verts(math.radians(iang))
        local = (verts + np.array([ix - ox, iy - oy])) @ R.T
        projs = local @ _NORMALS.T
        all_projs.append(projs)
    P = np.concatenate([p.ravel() for p in all_projs])
    if P.size == 0:
        return 0.0
    m = P.max()
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

def overlap_penalty(inner, min_dist=SQRT3 - 0.002):
    pen = 0.0
    n = len(inner)
    for i in range(n):
        xi, yi, _ = inner[i]
        for j in range(i + 1, n):
            xj, yj, _ = inner[j]
            d = math.hypot(xi - xj, yi - yj)
            if d < min_dist:
                pen += (min_dist - d) ** 2
    return pen

def build_lattice_config(axial_coords, s=1.0 + 1e-8):
    centers = [(1.5 * s * q, SQRT3 * s * (r + q / 2.0)) for q, r in axial_coords]
    return [[x, y, 0.0] for x, y in centers]

def rigid_optimize_orientation(centers, deadline):
    """Optimize global rotation of a rigid cluster (all hexagons aligned).
    Outer hexagon centered at centroid. 1D problem over angle in [0, 60)."""
    centers = np.array(centers, dtype=float)
    centroid = centers.mean(axis=0)
    rel = centers - centroid  # (11, 2)
    # All vertices relative to centroid: (11, 6, 2)
    verts_rel = rel[:, None, :] + _UNIT_VERTS[None, :, :]
    # Project onto normals: (11, 6, 6)
    proj0 = verts_rel @ _NORMALS.T  # (11,6,6) at angle 0
    max0 = proj0.max(axis=(0, 1))  # (6,)
    L0 = 2.0 * max(0.0, max0.max()) / SQRT3

    def L_of_angle(alpha_deg):
        a = math.radians(alpha_deg)
        c = math.cos(a); s = math.sin(a)
        # Rotate each vertex: R @ v
        # verts_rel shape (11,6,2); R = [[c,-s],[s,c]]
        # rotated = verts_rel @ R.T
        vr = verts_rel @ np.array([[c, s], [-s, c]])  # (11,6,2)
        proj = vr @ _NORMALS.T  # (11,6,6)
        m = proj.max()
        if m < 0:
            m = 0.0
        return 2.0 * m / SQRT3

    # Grid search for global minimum
    best_L = L0
    best_angle = 0.0
    n_grid = 600
    for i in range(n_grid):
        ang = i * 60.0 / n_grid
        L = L_of_angle(ang)
        if L < best_L:
            best_L = L
            best_angle = ang
    # Refine around best with golden section
    try:
        res = minimize_scalar(L_of_angle, bounds=(best_angle - 3.0, best_angle + 3.0),
                              method='bounded', options={'xatol': 1e-12})
        if res.fun < best_L:
            best_L = res.fun
            best_angle = float(res.x)
    except Exception:
        pass
    # Second refinement
    try:
        res = minimize_scalar(L_of_angle, bounds=(best_angle - 1.0, best_angle + 1.0),
                              method='bounded', options={'xatol': 1e-14})
        if res.fun < best_L:
            best_L = res.fun
            best_angle = float(res.x)
    except Exception:
        pass

    # Build inner config
    a = math.radians(best_angle)
    c = math.cos(a); s = math.sin(a)
    R = np.array([[c, -s], [s, c]])
    inner = []
    for i in range(len(centers)):
        inner.append([float(centroid[0] + rel[i, 0] * c - rel[i, 1] * s),
                      float(centroid[0] * 0 + centroid[1] + rel[i, 0] * s + rel[i, 1] * c),
                      float(best_angle)])
    # Fix centroid reference
    inner = []
    for i in range(len(centers)):
        px = rel[i, 0] * c - rel[i, 1] * s
        py = rel[i, 0] * s + rel[i, 1] * c
        inner.append([float(centroid[0] + px), float(centroid[1] + py), float(best_angle)])
    return inner, (float(centroid[0]), float(centroid[1])), float(best_angle), float(best_L)

def local_relax(inner, outer_center, outer_angle_deg, deadline):
    """Local relaxation to break rigid constraint. Optimize all DOFs."""
    def obj(p):
        il = []
        for i in range(N):
            il.append([p[i * 3], p[i * 3 + 1], p[i * 3 + 2]])
        ocx, ocy, oa = p[N * 3], p[N * 3 + 1], p[N * 3 + 2]
        side = compute_required_side_smooth(il, (ocx, ocy), oa, beta=50.0)
        pen = overlap_penalty(il)
        return side + 1000.0 * pen

    p0 = np.zeros(N * 3 + 3)
    for i, (x, y, a) in enumerate(inner):
        p0[i * 3] = x
        p0[i * 3 + 1] = y
        p0[i * 3 + 2] = a
    p0[N * 3] = outer_center[0]
    p0[N * 3 + 1] = outer_center[1]
    p0[N * 3 + 2] = outer_angle_deg

    best_p = p0.copy()
    best_v = obj(best_p)
    np.random.seed(42)
    for trial in range(3):
        if time.time() > deadline - 5:
            break
        p_start = best_p.copy() + np.random.normal(0, 0.02, size=best_p.shape)
        try:
            res = minimize(obj, p_start, method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12,
                                    'maxfev': 3000, 'adaptive': True})
            if res.fun < best_v:
                best_v = res.fun
                best_p = res.x.copy()
        except Exception:
            pass
    return best_p

def final_polish(params, deadline):
    """High-precision polish: smooth NM -> exact NM -> L-BFGS-B -> tight NM."""
    def obj_polish(p):
        il = []
        for i in range(N):
            il.append([p[i * 3], p[i * 3 + 1], p[i * 3 + 2]])
        ocx, ocy, oa = p[N * 3], p[N * 3 + 1], p[N * 3 + 2]
        side = compute_required_side_fast(il, (ocx, ocy), oa)
        pen = overlap_penalty(il)
        return side + 8000.0 * pen

    def obj_smooth(p):
        il = []
        for i in range(N):
            il.append([p[i * 3], p[i * 3 + 1], p[i * 3 + 2]])
        ocx, ocy, oa = p[N * 3], p[N * 3 + 1], p[N * 3 + 2]
        side = compute_required_side_smooth(il, (ocx, ocy), oa, beta=50.0)
        pen = overlap_penalty(il)
        return side + 1000.0 * pen

    best_p = params.copy()
    best_v = obj_polish(best_p)

    if time.time() < deadline - 6:
        np.random.seed(777)
        for trial in range(3):
            if time.time() > deadline - 6:
                break
            p_start = best_p.copy() + np.random.normal(0, 0.005, size=best_p.shape)
            try:
                res = minimize(obj_smooth, p_start, method='Nelder-Mead',
                               options={'maxiter': 1500, 'xatol': 1e-11, 'fatol': 1e-13,
                                        'maxfev': 2500, 'adaptive': True})
                if res.fun < best_v:
                    best_v = res.fun
                    best_p = res.x.copy()
            except Exception:
                pass

    if time.time() < deadline - 4:
        try:
            res = minimize(obj_polish, best_p, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-13, 'fatol': 1e-15,
                                    'maxfev': 4000, 'adaptive': True})
            if res.fun < best_v:
                best_v = res.fun
                best_p = res.x.copy()
        except Exception:
            pass

    if time.time() < deadline - 2:
        cx = sum(best_p[i * 3] for i in range(N)) / N
        cy = sum(best_p[i * 3 + 1] for i in range(N)) / N
        bounds = []
        for i in range(N):
            bounds.append((best_p[i * 3] - 0.3, best_p[i * 3] + 0.3))
            bounds.append((best_p[i * 3 + 1] - 0.3, best_p[i * 3 + 1] + 0.3))
            bounds.append((-180.0, 180.0))
        bounds.append((cx - 1.5, cx + 1.5))
        bounds.append((cy - 1.5, cy + 1.5))
        bounds.append((-30.0, 30.0))
        try:
            res = minimize(obj_polish, best_p, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 1500, 'ftol': 1e-16,
                                                   'gtol': 1e-13, 'maxfun': 4000})
            if res.fun < best_v:
                best_v = res.fun
                best_p = res.x.copy()
        except Exception:
            pass

    if time.time() < deadline - 1:
        try:
            res = minimize(obj_polish, best_p, method='Nelder-Mead',
                           options={'maxiter': 2500, 'xatol': 1e-14, 'fatol': 1e-16,
                                    'maxfev': 5000})
            if res.fun < best_v:
                best_v = res.fun
                best_p = res.x.copy()
        except Exception:
            pass

    return best_p

def construct():
    t0 = time.time()
    deadline = t0 + 145.0

    # Generate many 11-cell polyhex topologies
    axial1 = [(0, 0),
              (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
              (2, -1), (1, 1), (-2, 1), (-1, -1)]
    axial2 = [(-1, -1), (0, -1), (1, -1), (2, -1),
              (-1, 0), (0, 0), (1, 0),
              (-1, 1), (0, 1), (1, 1), (2, 1)]
    axial3 = [(-1, -1), (0, -1), (1, -1),
              (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
              (-1, 1), (0, 1), (1, 1)]
    axial4 = [(-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1),
              (-1, 0), (0, 0), (1, 0),
              (-1, 1), (0, 1), (1, 1)]
    axial5 = [(0, 0),
              (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
              (2, 0), (0, 2), (-2, 1), (-2, -1)]
    axial6 = [(0, -2), (1, -2),
              (-1, -1), (0, -1), (1, -1), (2, -1),
              (-1, 0), (0, 0), (1, 0),
              (0, 1), (1, 1)]
    axial7 = [(-1, -1), (0, -1), (1, -1),
              (-2, 0), (-1, 0), (0, 0), (1, 0),
              (-1, 1), (0, 1), (1, 1), (2, 1)]
    axial8 = [(0, 0), (1, 0), (2, 0), (3, 0),
              (-1, 1), (0, 1), (1, 1), (2, 1),
              (-1, -1), (0, -1), (1, -1)]
    axial9 = [(0, 0), (1, 0), (2, 0),
              (-1, 1), (0, 1), (1, 1), (2, 1), (3, 1),
              (-1, -1), (0, -1), (1, -1)]
    axial10 = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
               (0, 1), (1, 1), (2, 1),
               (0, -1), (1, -1), (2, -1)]
    axial11 = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
               (2, -1), (2, 0), (2, 1), (1, 1)]
    axial12 = [(0, 0), (1, 0), (2, 0), (3, 0),
               (1, 1), (2, 1), (3, 1),
               (1, -1), (2, -1), (3, -1)]
    axial13 = [(-1, 0), (0, 0), (1, 0), (2, 0),
               (-2, 1), (-1, 1), (0, 1), (1, 1),
               (-2, -1), (-1, -1), (0, -1)]
    axial14 = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
               (2, -1), (1, -2), (0, -2), (-1, -1)]

    configs = [axial1, axial2, axial3, axial4, axial5, axial6, axial7,
               axial8, axial9, axial10, axial11, axial12, axial13, axial14]

    best_side = float('inf')
    best_inner = None
    best_outer_center = None
    best_outer_angle = 0.0
    best_params = None

    # Phase 1: Rigid-body 1D optimization for each topology
    for ci, axial in enumerate(configs):
        if time.time() > deadline - 60:
            break
        s0 = 1.0 + 1e-8
        centers = [(1.5 * s0 * q, SQRT3 * s0 * (r + q / 2.0)) for q, r in axial]
        init_inner = [[x, y, 0.0] for x, y in centers]

        inner_r, oc_r, oa_r, side_r = rigid_optimize_orientation(centers, deadline)

        if side_r < best_side:
            best_side = side_r
            best_inner = inner_r
            best_outer_center = oc_r
            best_outer_angle = oa_r
            p0 = np.zeros(N * 3 + 3)
            for i, (x, y, a) in enumerate(inner_r):
                p0[i * 3] = x
                p0[i * 3 + 1] = y
                p0[i * 3 + 2] = a
            p0[N * 3] = oc_r[0]
            p0[N * 3 + 1] = oc_r[1]
            p0[N * 3 + 2] = oa_r
            best_params = p0.copy()

    if best_inner is None:
        s0 = 1.0 + 1e-8
        best_inner = build_lattice_config(configs[0], s0)
        cx0 = sum(c[0] for c in best_inner) / len(best_inner)
        cy0 = sum(c[1] for c in best_inner) / len(best_inner)
        best_outer_center = (cx0, cy0)
        best_outer_angle = 0.0
        best_side = compute_required_side_fast(best_inner, best_outer_center, best_outer_angle)
        p0 = np.zeros(N * 3 + 3)
        for i, (x, y, a) in enumerate(best_inner):
            p0[i * 3] = x
            p0[i * 3 + 1] = y
            p0[i * 3 + 2] = a
        p0[N * 3] = cx0
        p0[N * 3 + 1] = cy0
        p0[N * 3 + 2] = 0.0
        best_params = p0.copy()

    # Phase 2: Local relaxation to break rigid constraint
    if time.time() < deadline - 25 and best_params is not None:
        relaxed_params = local_relax(best_inner, best_outer_center, best_outer_angle, deadline)
        relaxed_inner = []
        for i in range(N):
            relaxed_inner.append([float(relaxed_params[i * 3]),
                                  float(relaxed_params[i * 3 + 1]),
                                  float(relaxed_params[i * 3 + 2])])
        relaxed_oc = (float(relaxed_params[N * 3]), float(relaxed_params[N * 3 + 1]))
        relaxed_oa = float(relaxed_params[N * 3 + 2])
        if check_disjoint(relaxed_inner, eps=1e-5):
            relaxed_side = compute_required_side_fast(relaxed_inner, relaxed_oc, relaxed_oa)
            if relaxed_side < best_side:
                best_inner = relaxed_inner
                best_outer_center = relaxed_oc
                best_outer_angle = relaxed_oa
                best_side = relaxed_side
                best_params = relaxed_params.copy()

    # Phase 3: Try local relaxation from second-best topology too
    if time.time() < deadline - 20 and best_params is not None:
        # Perturb and re-relax
        np.random.seed(999)
        p_pert = best_params.copy()
        for i in range(N):
            p_pert[i * 3] += np.random.normal(0, 0.05)
            p_pert[i * 3 + 1] += np.random.normal(0, 0.05)
            p_pert[i * 3 + 2] += np.random.normal(0, 5.0)
        p_pert[N * 3 + 2] += np.random.normal(0, 3.0)
        relaxed2 = local_relax(
            [[float(p_pert[i*3]), float(p_pert[i*3+1]), float(p_pert[i*3+2])] for i in range(N)],
            (float(p_pert[N*3]), float(p_pert[N*3+1])),
            float(p_pert[N*3+2]), deadline)
        r_inner = []
        for i in range(N):
            r_inner.append([float(relaxed2[i * 3]), float(relaxed2[i * 3 + 1]), float(relaxed2[i * 3 + 2])])
        r_oc = (float(relaxed2[N * 3]), float(relaxed2[N * 3 + 1]))
        r_oa = float(relaxed2[N * 3 + 2])
        if check_disjoint(r_inner, eps=1e-5):
            r_side = compute_required_side_fast(r_inner, r_oc, r_oa)
            if r_side < best_side:
                best_inner = r_inner
                best_outer_center = r_oc
                best_outer_angle = r_oa
                best_side = r_side
                best_params = relaxed2.copy()

    # Phase 4: High-precision final polish
    if time.time() < deadline - 8 and best_params is not None:
        polished = final_polish(best_params, deadline)
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

    # Fallback
    if not check_disjoint(best_inner, eps=1e-5):
        s0 = 1.0 + 1e-8
        best_inner = build_lattice_config(configs[0], s0)
        cx0 = sum(c[0] for c in best_inner) / len(best_inner)
        cy0 = sum(c[1] for c in best_inner) / len(best_inner)
        best_outer_center = (cx0, cy0)
        best_outer_angle = 0.0
        best_side = compute_required_side_fast(best_inner, best_outer_center, best_outer_angle)

    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
