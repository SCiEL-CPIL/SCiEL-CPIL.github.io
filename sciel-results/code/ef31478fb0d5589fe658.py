# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    ang = a + np.deg2rad(60 * np.arange(6))
    return np.stack([cx + s * np.cos(ang), cy + s * np.sin(ang)], axis=1)


def _hex_normals(a_deg):
    a = np.deg2rad(a_deg)
    base = a + np.deg2rad(60 * np.arange(6))
    normals = base + np.deg2rad(30)
    dirs = np.stack([np.cos(normals), np.sin(normals)], axis=1)
    return np.vstack([dirs, -dirs])


def _separation_ok(v1, n1, v2, n2):
    for d in n1:
        p1 = v1 @ d
        p2 = v2 @ d
        gap = min(p1) - max(p2)
        if gap > 1e-6:
            return True
        gap = min(p2) - max(p1)
        if gap > 1e-6:
            return True
    for d in n2:
        p1 = v1 @ d
        p2 = v2 @ d
        gap = min(p1) - max(p2)
        if gap > 1e-6:
            return True
        gap = min(p2) - max(p1)
        if gap > 1e-6:
            return True
    return False


def _hexagons_separated(inner_data):
    verts = []
    norms = []
    for i in range(len(inner_data)):
        x, y, a = inner_data[i]
        v = _hex_vertices(x, y, 1.0, a)
        n = _hex_normals(a)
        verts.append(v)
        norms.append(n)
    for i in range(len(inner_data)):
        for j in range(i + 1, len(inner_data)):
            if not _separation_ok(verts[i], norms[i], verts[j], norms[j]):
                return False
    return True


def _contains_in_hex(inner_data, outer_data, side):
    ox, oy, oa = outer_data
    outer_v = _hex_vertices(ox, oy, side, oa)
    tol = -1e-6
    for i in range(len(inner_data)):
        x, y, a = inner_data[i]
        v = _hex_vertices(x, y, 1.0, a)
        for p in v:
            for e in range(6):
                p1 = outer_v[e]
                p2 = outer_v[(e + 1) % 6]
                cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross < tol:
                    return False
    return True


def _valid(inner_data, outer_data, side):
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False
    if side < np.sqrt(11) - 1e-12:
        return False
    if not _hexagons_separated(inner_data):
        return False
    if not _contains_in_hex(inner_data, outer_data, side):
        return False
    for i in range(len(inner_data)):
        x, y, a = inner_data[i]
        v = _hex_vertices(x, y, 1.0, a)
        for e in range(6):
            p1 = v[e]
            p2 = v[(e + 1) % 6]
            if abs(np.linalg.norm(p2 - p1) - 1.0) > 1e-8:
                return False
    return True


def _compute_min_side(inner, outer_angle_deg):
    """Compute minimum outer hex side to contain all inner hexagons."""
    a_rad = np.deg2rad(outer_angle_deg)
    normals = []
    for k in range(6):
        ang = a_rad + np.deg2rad(30 + 60 * k)
        normals.append([np.cos(ang), np.sin(ang)])
    normals = np.array(normals)
    
    max_proj = 0.0
    for i in range(len(inner)):
        x, y, a = inner[i]
        v = _hex_vertices(x, y, 1.0, a)
        projs = np.abs(v @ normals.T)
        max_proj = max(max_proj, np.max(projs))
    
    return max_proj / (np.sqrt(3) / 2)


def _build_inner_symmetric(r1, r2, gangle):
    """Build 11 hexagons with 6-fold symmetry: 1 center + 6 ring + 4 outer."""
    inner = [[0.0, 0.0, gangle]]
    for k in range(6):
        ang = np.deg2rad(60 * k)
        x = r1 * np.cos(ang)
        y = r1 * np.sin(ang)
        inner.append([x, y, gangle])
    for k in [0, 2, 3, 5]:
        ang = np.deg2rad(60 * k + 30)
        x = r2 * np.cos(ang)
        y = r2 * np.sin(ang)
        inner.append([x, y, gangle])
    return np.array(inner)


def _min_separation_gap(inner_data):
    """Compute the minimum separation gap between all pairs of inner hexagons.
    Returns a positive value if all are separated, negative if any overlap."""
    verts = []
    norms = []
    for i in range(len(inner_data)):
        x, y, a = inner_data[i]
        v = _hex_vertices(x, y, 1.0, a)
        n = _hex_normals(a)
        verts.append(v)
        norms.append(n)
    
    min_gap = float('inf')
    for i in range(len(inner_data)):
        for j in range(i + 1, len(inner_data)):
            max_gap = -float('inf')
            for d in list(norms[i]) + list(norms[j]):
                p1 = verts[i] @ d
                p2 = verts[j] @ d
                gap = min(p1) - max(p2)
                gap2 = min(p2) - max(p1)
                max_gap = max(max_gap, gap, gap2)
            min_gap = min(min_gap, max_gap)
    return min_gap


def _smooth_objective(params):
    """Smooth objective: outer side + soft penalties for separation and containment."""
    r1, r2, gangle = params
    if r1 < 1.5 or r2 < r1 * 0.8 or r1 > 10 or r2 > 10:
        return 100.0 + abs(r1) + abs(r2)
    
    inner = _build_inner_symmetric(r1, r2, gangle)
    side = _compute_min_side(inner, gangle)
    
    # Separation penalty: smooth penalty if min gap is too small
    min_gap = _min_separation_gap(inner)
    if min_gap < 1e-4:
        sep_penalty = max(0.0, 1e-4 - min_gap) ** 2 * 100.0
    else:
        sep_penalty = 0.0
    
    # Containment check
    outer = np.array([0.0, 0.0, gangle])
    if not _contains_in_hex(inner, outer, side):
        ox, oy, oa = outer
        outer_v = _hex_vertices(ox, oy, side, oa)
        max_violation = 0.0
        for i in range(len(inner)):
            x, y, a = inner[i]
            v = _hex_vertices(x, y, 1.0, a)
            for p in v:
                for e in range(6):
                    p1 = outer_v[e]
                    p2 = outer_v[(e + 1) % 6]
                    cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                    if cross < 0:
                        max_violation = max(max_violation, -cross)
        cont_penalty = max_violation ** 2 * 100.0
    else:
        cont_penalty = 0.0
    
    return side + sep_penalty + cont_penalty


def _objective(params):
    """Objective: minimize outer hex side length with validity check.
    params = [r1, r2, gangle_deg]
    """
    r1, r2, gangle = params
    if r1 < 1.5 or r2 < r1 * 0.8 or r1 > 10 or r2 > 10:
        return 100.0 + abs(r1) + abs(r2)
    
    inner = _build_inner_symmetric(r1, r2, gangle)
    side = _compute_min_side(inner, gangle)
    
    outer = np.array([0.0, 0.0, gangle])
    if not _valid(inner, outer, side):
        return side + 50.0
    
    return side


def _polish(params):
    """High-precision polish pass starting from given params."""
    try:
        from scipy.optimize import minimize
        res = minimize(
            _smooth_objective,
            params,
            method='Nelder-Mead',
            options={'maxiter': 20000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
        )
        if res.fun < 10.0:
            r1, r2, gangle = res.x
            inner = _build_inner_symmetric(r1, r2, gangle)
            side = _compute_min_side(inner, gangle)
            outer = np.array([0.0, 0.0, gangle])
            if _valid(inner, outer, side):
                return res.x, side
    except Exception:
        pass
    return None, None


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns (inner_hex_data, outer_hex_data, outer_hex_side_length).
    """
    t0 = time.time()
    deadline = t0 + 610.0
    
    best_side = float('inf')
    best_inner = None
    best_outer = None
    best_params = None
    
    # Phase 1: Coarse search with smooth objective
    initial_guesses = [
        [2.5, 4.5, 0.0],
        [2.5, 4.5, 30.0],
        [2.6, 4.6, 0.0],
        [2.4, 4.4, 0.0],
        [2.5, 4.8, 15.0],
        [2.7, 4.7, 0.0],
        [2.5, 5.0, 0.0],
        [3.0, 5.0, 0.0],
        [2.5, 4.5, 60.0],
        [2.6, 4.9, 30.0],
        [2.8, 5.2, 0.0],
        [2.3, 4.2, 0.0],
        [2.9, 5.1, 15.0],
        [2.4, 4.6, 45.0],
        [2.7, 4.8, 30.0],
    ]
    
    for guess in initial_guesses:
        if time.time() > deadline - 90:
            break
        try:
            from scipy.optimize import minimize
            res = minimize(
                _smooth_objective,
                guess,
                method='Nelder-Mead',
                options={'maxiter': 15000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
            )
            if res.fun < 10.0:
                r1, r2, gangle = res.x
                inner = _build_inner_symmetric(r1, r2, gangle)
                side = _compute_min_side(inner, gangle)
                outer = np.array([0.0, 0.0, gangle])
                if _valid(inner, outer, side) and side < best_side:
                    best_side = side
                    best_params = res.x.copy()
        except Exception:
            continue
    
    # Phase 2: Polish the best solution found
    if best_params is not None and time.time() < deadline - 45:
        polished_params, polished_side = _polish(best_params)
        if polished_params is not None and polished_side < best_side:
            best_side = polished_side
            best_params = polished_params
    
    # Phase 3: Grid search for robustness
    if time.time() < deadline - 60:
        for r1 in np.arange(2.2, 3.2, 0.05):
            for r2 in np.arange(max(r1 + 0.3, 2.5), 5.5, 0.05):
                for ga in [0.0, 15.0, 30.0, 45.0]:
                    if time.time() > deadline - 30:
                        break
                    side = _smooth_objective([r1, r2, ga])
                    if side < best_side and side < 10.0:
                        inner = _build_inner_symmetric(r1, r2, ga)
                        side_check = _compute_min_side(inner, ga)
                        outer = np.array([0.0, 0.0, ga])
                        if _valid(inner, outer, side_check) and side_check < best_side:
                            best_side = side_check
                            best_params = np.array([r1, r2, ga])
    
    # Phase 4: Final polish
    if best_params is not None and time.time() < deadline - 15:
        polished_params, polished_side = _polish(best_params)
        if polished_params is not None and polished_side < best_side:
            best_side = polished_side
            best_params = polished_params
    
    if best_params is not None:
        r1, r2, gangle = best_params
        inner = _build_inner_symmetric(r1, r2, gangle)
        outer = np.array([0.0, 0.0, gangle])
        side = _compute_min_side(inner, gangle)
        
        if _valid(inner, outer, side):
            best_inner = inner
            best_outer = outer
            best_side = side
    
    if best_inner is None:
        # Fallback: known valid arrangement
        inner = np.array([
            [0, 0, 0],
            [-2.5, 0, 0],
            [2.5, 0, 0],
            [-1.25, 2.165, 0],
            [1.25, 2.165, 0],
            [-1.25, -2.165, 0],
            [1.25, -2.165, 0],
            [-3.75, 2.165, 0],
            [3.75, 2.165, 0],
            [-3.75, -2.165, 0],
            [3.75, -2.165, 0],
        ])
        outer = np.array([0, 0, 0])
        side = _compute_min_side(inner, 0.0)
        if not _valid(inner, outer, side):
            side = 8.0
            if not _valid(inner, outer, side):
                inner = np.array([
                    [0, 0, 0],
                    [-2.5, 0, 0],
                    [2.5, 0, 0],
                    [-1.25, 2.165, 0],
                    [1.25, 2.165, 0],
                    [-1.25, -2.165, 0],
                    [1.25, -2.165, 0],
                    [-3.75, 2.165, 0],
                    [3.75, 2.165, 0],
                    [-3.75, -2.165, 0],
                    [3.75, -2.165, 0],
                ])
                outer = np.array([0, 0, 0])
                side = 9.0
        
        return inner, outer, side
    
    return best_inner, best_outer, best_side


# EVOLVE-BLOCK-END
