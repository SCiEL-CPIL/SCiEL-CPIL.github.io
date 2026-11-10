# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

_START_TIME = time.time()
_DEADLINE = 630.0
_SAFETY = 20.0


def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START_TIME)


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    angs = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angs), cy + s * np.sin(angs)])


def _edge_normals(verts):
    n = []
    m = len(verts)
    for i in range(m):
        p1 = verts[i]
        p2 = verts[(i + 1) % m]
        e = p2 - p1
        L = np.linalg.norm(e)
        if L < 1e-15:
            continue
        nx = (p2[1] - p1[1]) / L
        ny = -(p2[0] - p1[0]) / L
        n.append([nx, ny])
        n.append([-nx, -ny])
    return np.array(n)


def _hexagons_separated(v1, v2):
    n1 = _edge_normals(v1)
    n2 = _edge_normals(v2)
    for N in (n1, n2):
        for row in N:
            proj1 = v1 @ row
            proj2 = v2 @ row
            if proj1.max() - proj2.min() < -1e-6:
                return True
            if proj2.max() - proj1.min() < -1e-6:
                return True
    return False


def _all_inner_separated(inner_verts):
    n = len(inner_verts)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_separated(inner_verts[i], inner_verts[j]):
                return False
    return True


def _inner_in_outer(inner_verts, outer_verts):
    for iv in inner_verts:
        m = len(outer_verts)
        for i in range(m):
            p1 = outer_verts[i]
            p2 = outer_verts[(i + 1) % m]
            cross = np.cross(p2 - p1, iv - p1)
            if cross < -1e-6:
                return False
    return True


def _enclosing_side(inner):
    """Minimal side of a regular hexagon (centered at origin, vertex at 0 deg)
    containing all inner unit hexagons. Uses exact 6-half-plane definition."""
    normals = []
    for ang in (30, 90, 150, 210, 270, 330):
        a = np.deg2rad(ang)
        normals.append([np.cos(a), np.sin(a)])
    normals = np.array(normals)
    apothem_needed = 0.0
    for row in inner:
        iv = _hex_vertices(row[0], row[1], 1.0, row[2])
        for p in iv:
            for nrm in normals:
                val = p @ nrm
                if val > apothem_needed:
                    apothem_needed = val
    return apothem_needed * 2.0 / np.sqrt(3.0)


def _valid_configuration(inner, outer, side):
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False, 0.0
    if side < np.sqrt(11.0) - 1e-9:
        return False, 0.0
    outer_verts = _hex_vertices(outer[0], outer[1], side, outer[2])
    inner_verts = [_hex_vertices(r[0], r[1], 1.0, r[2]) for r in inner]
    if not _all_inner_separated(inner_verts):
        return False, 0.0
    if not _inner_in_outer(inner_verts, outer_verts):
        return False, 0.0
    for iv in inner_verts:
        for i in range(6):
            d = np.linalg.norm(iv[(i + 1) % 6] - iv[i])
            if abs(d - 1.0) > 1e-8:
                return False, 0.0
    return True, 1.0 / side


def _pair_overlap_penalty(v1, v2, w=100.0):
    n1 = _edge_normals(v1)
    n2 = _edge_normals(v2)
    pen = 0.0
    for N in (n1, n2):
        for row in N:
            proj1 = v1 @ row
            proj2 = v2 @ row
            gap = proj1.max() - proj2.min()
            if gap > -1e-6:
                pen += w * gap * gap
            gap2 = proj2.max() - proj1.min()
            if gap2 > -1e-6:
                pen += w * gap2 * gap2
    return pen


def _containment_penalty(inner_verts, outer_verts, w=200.0):
    pen = 0.0
    m = len(outer_verts)
    for iv in inner_verts:
        for p in iv:
            for i in range(m):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % m]
                cross = np.cross(p2 - p1, p - p1)
                if cross < 0:
                    pen += w * cross * cross
    return pen


def _symmetric_packing(r1, r2):
    inner = [[0.0, 0.0, 0.0]]
    for ang in (90.0, 210.0, 330.0):
        a = np.deg2rad(ang)
        inner.append([r1 * np.cos(a), r1 * np.sin(a), 60.0])
    for ang in (30.0, 150.0, 270.0):
        a = np.deg2rad(ang)
        inner.append([r2 * np.cos(a), r2 * np.sin(a), 120.0])
    inner = np.array(inner)
    side = _enclosing_side(inner)
    outer = np.array([0.0, 0.0, 0.0])
    return inner, outer, side


def _symmetric_objective(x):
    r1, r2 = x
    if r1 < 0 or r2 < 0:
        return 1e9
    inner, outer, side = _symmetric_packing(r1, r2)
    if side <= 0:
        return 1e9
    inner_verts = [_hex_vertices(r[0], r[1], 1.0, r[2]) for r in inner]
    outer_verts = _hex_vertices(0, 0, side, 0)
    pen = _containment_penalty(inner_verts, outer_verts)
    n = len(inner_verts)
    for i in range(n):
        for j in range(i + 1, n):
            pen += _pair_overlap_penalty(inner_verts[i], inner_verts[j])
    return side + pen


def _full_objective(params, side):
    inner = params.reshape(11, 3)
    inner_verts = [_hex_vertices(r[0], r[1], 1.0, r[2]) for r in inner]
    outer_verts = _hex_vertices(0, 0, side, 0)
    pen = _containment_penalty(inner_verts, outer_verts)
    n = len(inner_verts)
    for i in range(n):
        for j in range(i + 1, n):
            pen += _pair_overlap_penalty(inner_verts[i], inner_verts[j])
    return pen


def _refine_full(inner, side):
    x0 = inner.ravel()

    def obj(x):
        if _remaining() < 5:
            return 1e9
        return _full_objective(x, side)

    try:
        res = minimize(obj, x0, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8,
                                'adaptive': True})
        x = res.x
    except Exception:
        x = x0
    return x.reshape(11, 3)


def hexagon_packing_11():
    best_inner = None
    best_outer = None
    best_side = np.inf
    best_val = 0.0

    def try_candidate(inner, outer, side):
        nonlocal best_inner, best_outer, best_side, best_val
        valid, val = _valid_configuration(inner, outer, side)
        if valid and val > best_val:
            best_inner = inner.copy()
            best_outer = outer.copy()
            best_side = side
            best_val = val

    # Phase 1: symmetric optimization
    if _remaining() > 30:
        x0 = np.array([1.8, 3.4])
        try:
            res = minimize(_symmetric_objective, x0, method='Nelder-Mead',
                           options={'maxiter': 2500, 'xatol': 1e-5, 'fatol': 1e-7,
                                    'adaptive': True})
            r1, r2 = res.x
            inner, outer, side = _symmetric_packing(r1, r2)
            try_candidate(inner, outer, side)
        except Exception:
            pass
        for (r1, r2) in [(1.732, 3.464), (1.8, 3.5), (1.6, 3.2), (2.0, 3.8), (1.9, 3.6), (1.7, 3.3)]:
            inner, outer, side = _symmetric_packing(r1, r2)
            try_candidate(inner, outer, side)

    # Phase 2: full 33-DOF refinement
    if best_inner is not None and _remaining() > 40:
        refined = _refine_full(best_inner, best_side)
        side = _enclosing_side(refined)
        try_candidate(refined, np.array([0.0, 0.0, 0.0]), side)
        for _ in range(4):
            if _remaining() < 30:
                break
            refined = _refine_full(refined, side)
            side = _enclosing_side(refined)
            try_candidate(refined, np.array([0.0, 0.0, 0.0]), side)

    # Fallback valid config
    if best_inner is None:
        inner = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.17, 0],
            [1.25, 2.17, 0],
            [-1.25, -2.17, 0],
            [1.25, -2.17, 0],
            [-3.75, 2.17, 0],
            [3.75, 2.17, 0],
            [-3.75, -2.17, 0],
            [3.75, -2.17, 0],
        ], dtype=float)
        outer = np.array([0.0, 0.0, 0.0])
        side = 8.0
        try_candidate(inner, outer, side)

    return best_inner, best_outer, float(best_side)


# EVOLVE-BLOCK-END
