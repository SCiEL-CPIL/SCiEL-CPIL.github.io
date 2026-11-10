# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 600.0
_EPS = 1e-9


def _hex_vertices(cx, cy, s, a_rad):
    t = a_rad + np.arange(6) * (np.pi / 3.0)
    return np.column_stack((cx + s * np.cos(t), cy + s * np.sin(t)))


def _pair_sep(verts1, verts2):
    """Max over 12 edge normals of the signed separation between two convex hexagons.
    Positive => separated by that amount on some normal; negative => overlap depth."""
    minsep = -np.inf
    for V in (verts1, verts2):
        for i in range(6):
            e = V[(i + 1) % 6] - V[i]
            n = np.array([-e[1], e[0]])
            nl = np.hypot(*n)
            if nl < 1e-12:
                continue
            n = n / nl
            p1 = V[i] @ n
            p2 = V[(i + 1) % 6] @ n
            lo1, hi1 = min(p1, p2), max(p1, p2)
            proj2 = verts2 @ n
            lo2, hi2 = proj2.min(), proj2.max()
            sep = min(lo2 - hi1, lo1 - hi2)
            if sep > minsep:
                minsep = sep
    return minsep


def _sep_matrix(verts_list):
    n = len(verts_list)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            s = _pair_sep(verts_list[i], verts_list[j])
            M[i, j] = s
            M[j, i] = s
    return M


def _min_pair_sep(verts_list):
    M = _sep_matrix(verts_list)
    return M[np.triu_indices(len(verts_list), 1)].min()


def _outer_min_side(centers, angles, s_inner, n_grid=1200):
    """Smallest side length of a regular hexagon (centered at origin) that contains all
    inner hexagons, sweeping the outer orientation. Returns (best_side, best_angle_deg)."""
    best = np.inf
    best_a = 0.0
    for a in np.linspace(0.0, np.pi / 3.0, n_grid, endpoint=False):
        # outer vertices (unit side, centered at origin)
        V = _hex_vertices(0.0, 0.0, 1.0, a)
        # edge inward normals
        normals = []
        for i in range(6):
            e = V[(i + 1) % 6] - V[i]
            n = np.array([-e[1], e[0]])
            nl = np.hypot(*n)
            n = n / nl
            # make it inward (pointing toward origin)
            if np.dot(n, V[i]) > 0:
                n = -n
            normals.append(n)
        normals = np.array(normals)
        # support of each inner hexagon along each normal
        req = np.zeros((6, len(centers)))
        for k, (cx, cy, ang) in enumerate(zip(centers, angles)):
            verts = _hex_vertices(cx, cy, s_inner, ang)
            proj = verts @ normals.T  # (6,6)
            req[:, k] = proj.max(axis=1)
        # side length needed: scale so that req <= apothem = side * sqrt(3)/2
        # => side >= req / (sqrt(3)/2) = req * 2/sqrt(3)
        side = (req.max() * 2.0) / np.sqrt(3.0)
        if side < best:
            best = side
            best_a = a
    return best, np.degrees(best_a)


def _validity(centers, angles, s_inner, side, outer_angle_deg, tol=1e-6):
    """Check all constraints. Returns True if valid."""
    if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(angles)):
        return False
    if side < np.sqrt(11.0) - 1e-9:
        return False
    verts_list = [_hex_vertices(cx, cy, s_inner, np.radians(ang))
                  for cx, cy, ang in zip(centers, angles)]
    # pairwise separation
    for i in range(len(verts_list)):
        for j in range(i + 1, len(verts_list)):
            if _pair_sep(verts_list[i], verts_list[j]) < -tol:
                return False
    # containment
    V = _hex_vertices(0.0, 0.0, side, np.radians(outer_angle_deg))
    for Vh in verts_list:
        for i in range(6):
            p1 = V[i]
            p2 = V[(i + 1) % 6]
            cross = np.cross(p2 - p1, Vh - p1)
            if cross.min() < -tol:
                return False
    # edge length of inner hexagons ~ 1
    for Vh in verts_list:
        for i in range(6):
            L = np.linalg.norm(Vh[(i + 1) % 6] - Vh[i])
            if abs(L - 1.0) > 1e-8:
                return False
    return True


def _honeycomb_layout(scale=1.0, rot=0.0):
    """Honeycomb arrangement of 11 unit hexagons (side=1), centered near origin."""
    dx = 2.0 * scale
    dy = np.sqrt(3.0) * scale
    pts = [
        (0, 0),
        (-dx, 0), (dx, 0),
        (-dx / 2, dy), (dx / 2, dy),
        (-dx / 2, -dy), (dx / 2, -dy),
        (-3 * dx / 2, dy), (3 * dx / 2, dy),
        (-3 * dx / 2, -dy), (3 * dx / 2, -dy),
    ]
    pts = np.array(pts, dtype=float)
    if rot != 0.0:
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s], [s, c]])
        pts = pts @ R.T
    centers = pts
    angles = np.zeros(len(pts))
    return centers, angles


def hexagon_packing_11():
    # ---- Initial honeycomb layout (unit side) ----
    centers, angles = _honeycomb_layout(scale=1.0, rot=0.0)
    s_inner = 1.0
    n = len(centers)

    # ---- Optimize positions to maximize minimum separation ----
    t_start = time.time()
    best_minsep = _min_pair_sep(
        [_hex_vertices(cx, cy, s_inner, np.radians(a)) for cx, cy, a in zip(centers, angles)]
    )
    best_centers, best_angles = centers.copy(), angles.copy()

    # Simple coordinate-descent / random-restart hill climbing on positions
    rng = np.random.default_rng(12345)
    step = 0.05
    for it in range(4000):
        if time.time() - t_start > _DEADLINE - 120.0:
            break
        # pick a random hexagon and nudge it
        idx = rng.integers(n)
        old = centers[idx].copy()
        centers[idx] = old + step * (rng.standard_normal(2) - 0.5)
        verts_list = [_hex_vertices(cx, cy, s_inner, np.radians(a))
                      for cx, cy, a in zip(centers, angles)]
        ms = _min_pair_sep(verts_list)
        if ms > best_minsep + 1e-12:
            best_minsep = ms
            best_centers, best_angles = centers.copy(), angles.copy()
        else:
            centers[idx] = old
        # occasional restart with different random layout
        if it % 500 == 499 and it < 2500:
            c2, a2 = _honeycomb_layout(scale=rng.uniform(0.9, 1.1),
                                       rot=rng.uniform(0, 2 * np.pi))
            v2 = [_hex_vertices(cx, cy, s_inner, np.radians(ang))
                  for cx, cy, ang in zip(c2, a2)]
            ms2 = _min_pair_sep(v2)
            if ms2 > best_minsep:
                best_minsep = ms2
                best_centers, best_angles = c2.copy(), a2.copy()
                centers, angles = c2.copy(), a2.copy()

    centers, angles = best_centers, best_angles

    # ---- Determine minimal enclosing side via outer-orientation sweep ----
    side, outer_angle_deg = _outer_min_side(centers, angles, s_inner, n_grid=800)

    # ---- Scale layout so it just fits: we want the packing to be valid at this side.
    #    The sweep already assumed the current (unit) layout; side is the required
    #    enclosing side for that exact layout. So we keep the layout as-is and report
    #    that side. But to be safe, scale inner hexagons down slightly if needed.
    #    Actually the layout is fixed unit hexagons; side is what contains them.
    #    We just need to ensure validity at (side, outer_angle_deg).
    #    If invalid (shouldn't be), shrink inner side a touch and recompute.
    for shrink in [1.0, 0.999, 0.998, 0.995, 0.99]:
        s_try = s_inner * shrink
        if _validity(centers, angles, s_try, side, outer_angle_deg):
            s_inner = s_try
            break

    # Final: recompute side with final inner side (in case shrunk)
    side, outer_angle_deg = _outer_min_side(centers, angles, s_inner, n_grid=1200)

    # Build return arrays
    inner_hex_data = np.column_stack((centers, angles))
    outer_hex_data = np.array([0.0, 0.0, outer_angle_deg])

    return inner_hex_data, outer_hex_data, float(side)


# EVOLVE-BLOCK-END
