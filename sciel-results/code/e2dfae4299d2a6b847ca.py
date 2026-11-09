# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _edge_normals(hex_verts):
    """6 outward unit edge normals for a hexagon given 6 ordered vertices."""
    normals = []
    centroid = hex_verts.mean(axis=0)
    for i in range(6):
        p1 = hex_verts[i]
        p2 = hex_verts[(i + 1) % 6]
        edge = p2 - p1
        n = np.array([-edge[1], edge[0]])
        mid = (p1 + p2) / 2.0
        if np.dot(n, mid - centroid) < 0:
            n = -n
        n = n / np.linalg.norm(n)
        normals.append(n)
    return np.array(normals)


def _separated(v1, n1, v2, n2):
    """True if two hexagons (vertices v1,v2; outward normals n1,n2) are apart by > 1e-6
    on at least one of their 12 edge normals."""
    all_normals = np.vstack([n1, n2])  # 12 x 2
    proj1 = v1 @ all_normals.T  # 6 x 12
    proj2 = v2 @ all_normals.T
    min1, max1 = proj1.min(axis=0), proj1.max(axis=0)
    min2, max2 = proj2.min(axis=0), proj2.max(axis=0)
    gaps = np.minimum(max1 - min2, max2 - min1)  # positive => separated along that normal
    return bool(np.any(gaps > 1e-6))


def _all_separated(inner_hexes):
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _separated(inner_hexes[i][0], inner_hexes[i][1],
                              inner_hexes[j][0], inner_hexes[j][1]):
                return False
    return True


def _fits(S, angle_deg, inner_hexes):
    """True if every inner vertex lies inside the outer hexagon (center 0, side S, angle)."""
    outer_verts = _hex_vertices(0.0, 0.0, S, angle_deg)
    for hex_verts, _ in inner_hexes:
        for p in hex_verts:
            for i in range(6):
                p1 = outer_verts[i]
                p2 = outer_verts[(i + 1) % 6]
                cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross < -1e-6:
                    return False
    return True


def _min_outer_side(inner_hexes):
    """Minimum (S, angle) of a centered outer hexagon containing all inner hexagons."""
    # Gather all inner vertices (relative to origin, since outer is centered at 0)
    all_pts = np.vstack([hv for hv, _ in inner_hexes])  # (66, 2)
    max_r = float(np.max(np.linalg.norm(all_pts, axis=1)))

    def required_side(angle_deg):
        """Smallest S such that all_pts fit an outer hexagon of side S at this angle."""
        outer_verts = _hex_vertices(0.0, 0.0, 1.0, angle_deg)  # unit side
        best = 1e-9
        for i in range(6):
            p1 = outer_verts[i]
            p2 = outer_verts[(i + 1) % 6]
            e = p2 - p1
            # cross(e, p - p1) >= 0  <=>  e.x*(p.y-p1.y) - e.y*(p.x-p1.x) >= 0
            val = e[0] * (all_pts[:, 1] - p1[1]) - e[1] * (all_pts[:, 0] - p1[0])
            # need S * val >= 0  ->  S >= -val for the most-negative val
            worst = float(val.min())
            if worst < 0.0:
                best = max(best, -worst)
        return best

    # Coarse sweep
    angles = np.arange(0.0, 60.0, 5.0)
    reqs = np.array([required_side(a) for a in angles])
    i0 = int(np.argmin(reqs))
    best_S = float(reqs[i0])
    best_angle = float(angles[i0])

    # Fine sweep around the best
    fine = best_angle + np.linspace(-5.0, 5.0, 41)
    for a in fine:
        r = required_side(a)
        if r < best_S:
            best_S = r
            best_angle = float(a)

    # Binary search on S at the best angle
    sl = max_r * 0.9
    sh = best_S
    for _ in range(30):
        sm = (sl + sh) / 2.0
        if _fits(sm, best_angle, inner_hexes):
            sh = sm
        else:
            sl = sm
    best_S = sh

    # Safety: verify the reported configuration actually fits; else enlarge
    if not _fits(best_S, best_angle, inner_hexes):
        S = max_r
        for _ in range(60):
            if _fits(S, best_angle, inner_hexes):
                break
            S *= 1.05
        best_S = S

    return best_S, best_angle


def hexagon_packing_11():
    t_start = time.time()
    deadline = 600.0
    safety = 120.0

    seed_positions = np.array(
        [
            [0.0, 0.0],
            [-2.5, 0.0],
            [2.5, 0.0],
            [-1.25, 2.1650635094610964],
            [1.25, 2.1650635094610964],
            [-1.25, -2.1650635094610964],
            [1.25, -2.1650635094610964],
            [-3.75, 2.1650635094610964],
            [3.75, 2.1650635094610964],
            [-3.75, -2.1650635094610964],
            [3.75, -2.1650635094610964],
        ]
    )

    def _build_inner_hexes(positions, angles):
        hexes = []
        for i in range(11):
            verts = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            norms = _edge_normals(verts)
            hexes.append((verts, norms))
        return hexes

    def _objective(params):
        if time.time() > deadline - safety:
            return 1e10
        positions = params[:22].reshape(11, 2)
        angles = params[22:]
        inner_hexes = _build_inner_hexes(positions, angles)
        if not _all_separated(inner_hexes):
            # mild penalty to steer away from overlaps
            return 1e10
        S, _ = _min_outer_side(inner_hexes)
        if not np.isfinite(S):
            return 1e10
        return S

    x0 = np.zeros(33)
    x0[:22] = seed_positions.flatten()
    x0[22:] = 0.0

    seed_hexes = _build_inner_hexes(seed_positions, np.zeros(11))
    best_S, best_angle = _min_outer_side(seed_hexes)
    best_params = x0.copy()

    # Local Nelder-Mead polish from the seed (cheap, bounded)
    if time.time() < deadline - safety:
        try:
            res = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True},
            )
            if res.fun < best_S and res.fun < 1e9:
                cand = res.x
                cpos = cand[:22].reshape(11, 2)
                can = cand[22:]
                chex = _build_inner_hexes(cpos, can)
                if _all_separated(chex):
                    s2, a2 = _min_outer_side(chex)
                    if s2 < best_S:
                        best_S, best_angle = s2, a2
                        best_params = cand
        except Exception:
            pass

    # Extra budget: a few more restarts / longer runs
    if time.time() < deadline - safety:
        try:
            res = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 800, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True},
            )
            if res.fun < best_S and res.fun < 1e9:
                cand = res.x
                cpos = cand[:22].reshape(11, 2)
                can = cand[22:]
                chex = _build_inner_hexes(cpos, can)
                if _all_separated(chex):
                    s2, a2 = _min_outer_side(chex)
                    if s2 < best_S:
                        best_S, best_angle = s2, a2
                        best_params = cand
        except Exception:
            pass

    positions = best_params[:22].reshape(11, 2)
    angles = best_params[22:]

    # Final verification; fall back to seed if anything is off
    final_hexes = _build_inner_hexes(positions, angles)
    if not _all_separated(final_hexes):
        positions = seed_positions.copy()
        angles = np.zeros(11)
        final_hexes = _build_inner_hexes(positions, angles)
        best_S, best_angle = _min_outer_side(final_hexes)

    inner_hex_data = np.column_stack([positions, angles])
    outer_hex_data = np.array([0.0, 0.0, best_angle])
    outer_hex_side_length = float(best_S)

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
