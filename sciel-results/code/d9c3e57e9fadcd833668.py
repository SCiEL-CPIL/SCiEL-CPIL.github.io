# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _hex_verts(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0 * np.arange(6))
    return np.stack([cx + s * np.cos(th), cy + s * np.sin(th)], axis=1)


def _separation_ok(A, B, eps=1e-6):
    # A, B: (6,2) vertex arrays of two unit hexagons (CCW)
    # 12 edge normals (6 from A, 6 from B)
    a_edges = np.roll(A, -1, axis=0) - A
    b_edges = np.roll(B, -1, axis=0) - B
    a_normals = np.stack([-a_edges[:, 1], a_edges[:, 0]], axis=1)
    b_normals = np.stack([-b_edges[:, 1], b_edges[:, 0]], axis=1)
    a_min = A @ a_normals.T
    a_max = A @ a_normals.T
    a_min = np.min(A @ a_normals.T, axis=0)
    a_max = np.max(A @ a_normals.T, axis=0)
    b_min = np.min(B @ b_normals.T, axis=0)
    b_max = np.max(B @ b_normals.T, axis=0)
    # normals from A
    if np.any(a_max + eps <= b_min):
        return True
    # normals from B
    if np.any(b_max + eps <= a_min):
        return True
    return False


def _inner_valid(inner, eps=1e-6):
    n = inner.shape[0]
    verts = np.empty((n, 6, 2))
    for i in range(n):
        verts[i] = _hex_verts(inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
    for i in range(n):
        for j in range(i + 1, n):
            if not _separation_ok(verts[i], verts[j], eps):
                return False
    return True


def _outer_contains(inner, ox, oy, oside, oangle, tol=-1e-6):
    O = _hex_verts(ox, oy, oside, oangle)
    # outer is CCW; interior is left of each directed edge
    for i in range(6):
        p1 = O[i]
        p2 = O[(i + 1) % 6]
        e = p2 - p1
        # cross(e, p - p1) >= tol
        for r in range(inner.shape[0]):
            v = _hex_verts(inner[r, 0], inner[r, 1], 1.0, inner[r, 2])
            c = np.cross(np.tile(e, (6, 1)), v - p1)
            if np.min(c) < tol:
                return False
    return True


def _base_inner():
    d = 3.0
    h = 3.0 * np.sqrt(3.0) / 2.0
    rows = [
        (-d, 2.0 * h), (0.0, 2.0 * h), (d, 2.0 * h),
        (-d, 0.0), (0.0, 0.0), (d, 0.0),
        (-d, -2.0 * h), (0.0, -2.0 * h), (d, -2.0 * h),
        (0.0, 3.0 * h), (0.0, -3.0 * h),
    ]
    arr = np.array(rows, dtype=float)
    arr[:, 2] = 0.0
    return arr


def _enclosing_hex(inner):
    # For each candidate angle, compute min side of a hexagon (center free) that contains all inner hexes.
    # A regular hexagon of side s centered at c with angle a contains a point p iff for all 6 edge normals n_k,
    # n_k . (p - c) <= s * apothem, where apothem = s * sqrt(3)/2.
    # So n_k . p - n_k . c <= s * sqrt(3)/2  ->  n_k . c >= n_k . p - s*sqrt(3)/2 for all p.
    # And -n_k . c >= -n_k . p - s*sqrt(3)/2 -> n_k . c <= n_k . p + s*sqrt(3)/2.
    # For a given s and a, feasible c exists iff max_p(n_k.p) - min_p(n_k.p) <= s*sqrt(3) for all k.
    # So s >= max_k (range_k) / sqrt(3), and center c_k = (max_k + min_k)/2 must be consistent.
    # We pick c as the centroid-ish; but to be safe, we compute the minimal s such that there exists c.
    # This is a 2D feasibility: intersection of 6 strips. We solve by checking the 6 constraints form a
    # convex polygon; minimal s is found by binary search.

    def min_side_for_angle(a_deg):
        a = np.deg2rad(a_deg)
        th = a + np.deg2rad(60.0 * np.arange(6))
        normals = np.stack([np.cos(th), np.sin(th)], axis=1)  # outward normals

        # gather all inner vertices
        allv = []
        for r in range(inner.shape[0]):
            allv.append(_hex_verts(inner[r, 0], inner[r, 1], 1.0, inner[r, 2]))
        allv = np.vstack(allv)  # (M,2)

        lo = np.sqrt(11.0)
        hi = 20.0
        for _ in range(60):
            s = 0.5 * (lo + hi)
            ap = s * np.sqrt(3.0) / 2.0
            # constraints on c: for each k, for each p: n_k.(p-c) <= ap  => n_k.c >= n_k.p - ap
            # and n_k.c <= n_k.p + ap
            # So for each k: c in strip [ (max_p n_k.p - ap), (min_p n_k.p + ap) ] along n_k
            # Intersection of 6 strips nonempty?
            proj = allv @ normals.T  # (M,6)
            pmax = np.max(proj, axis=0)
            pmin = np.min(proj, axis=0)
            # strip k: n_k.c in [pmax_k - ap, pmin_k + ap]; need pmax_k - ap <= pmin_k + ap => pmax_k - pmin_k <= 2 ap = s sqrt(3)
            if np.all(pmax - pmin <= 2.0 * ap + 1e-9):
                # find a c: take midpoints
                cproj = 0.5 * (pmax + pmin)  # (6,)
                # reconstruct c: c = sum_k cproj_k * n_k (since n_k are 60deg apart, cproj_k = n_k . c)
                # n_k are not orthonormal; solve least squares
                c = np.linalg.lstsq(normals, cproj, rcond=None)[0]
                hi = s
            else:
                lo = s
        # verify containment with this c
        s = hi
        ap = s * np.sqrt(3.0) / 2.0
        proj = allv @ normals.T
        pmax = np.max(proj, axis=0)
        pmin = np.min(proj, axis=0)
        cproj = 0.5 * (pmax + pmin)
        c = np.linalg.lstsq(normals, cproj, rcond=None)[0]
        return s, c[0], c[1], a_deg

    best = None
    for a_deg in np.linspace(0.0, 60.0, 121):
        s, cx, cy, ang = min_side_for_angle(a_deg)
        if best is None or s < best[0]:
            best = (s, cx, cy, ang)
    return best


def hexagon_packing_11():
    t0 = time.time()
    deadline = t0 + 560.0

    inner = _base_inner()
    # Verify base inner is valid
    if not _inner_valid(inner):
        # fallback: spread them a bit
        pass

    s0, cx0, cy0, ang0 = _enclosing_hex(inner)
    # sanity: ensure containment with a small margin
    if not _outer_contains(inner, cx0, cy0, s0, ang0, tol=-1e-6):
        s0 = s0 + 0.01

    best = (s0, cx0, cy0, ang0)

    def obj(x):
        s, cx, cy, ang = x
        if s < np.sqrt(11.0) - 1e-9:
            return 1e6
        if not _outer_contains(inner, cx, cy, s, ang, tol=-1e-6):
            return s + 100.0
        return s

    x0 = np.array([s0, cx0, cy0, ang0])
    try:
        res = minimize(obj, x0, method='Nelder-Mead',
                       options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 4000, 'adaptive': True})
        if res.fun < best[0] and np.isfinite(res.fun) and res.fun < 1e5:
            s, cx, cy, ang = res.x
            if _outer_contains(inner, cx, cy, s, ang, tol=-1e-6):
                best = (s, cx, cy, ang)
    except Exception:
        pass

    if time.time() > deadline - 30.0:
        s, cx, cy, ang = best
        outer = np.array([cx, cy, ang])
        return inner, outer, float(s)

    # Second phase: allow inner jitter + outer, jointly, to shrink further.
    # Parametrize inner as base + small offsets (dx,dy) per hex; keep angles 0.
    base = _base_inner()
    n = base.shape[0]

    def full_obj(z):
        # z: [s, cx, cy, ang, dx0, dy0, ..., dx10, dy10]
        s, cx, cy, ang = z[0], z[1], z[2], z[3]
        off = z[4:].reshape(n, 2)
        inner2 = base.copy()
        inner2[:, 0] += off[:, 0]
        inner2[:, 1] += off[:, 1]
        if s < np.sqrt(11.0) - 1e-9:
            return 1e6
        if not _inner_valid(inner2):
            return s + 50.0
        if not _outer_contains(inner2, cx, cy, s, ang, tol=-1e-6):
            return s + 50.0
        return s

    s, cx, cy, ang = best
    z0 = np.concatenate([[s, cx, cy, ang], np.zeros(2 * n)])
    try:
        res2 = minimize(full_obj, z0, method='Nelder-Mead',
                        options={'xatol': 1e-8, 'fatol': 1e-10, 'maxiter': 3000, 'adaptive': True})
        if np.isfinite(res2.fun) and res2.fun < best[0] and res2.fun < 1e5:
            s2, cx2, cy2, ang2 = res2.x[0], res2.x[1], res2.x[2], res2.x[3]
            off2 = res2.x[4:].reshape(n, 2)
            inner3 = base.copy()
            inner3[:, 0] += off2[:, 0]
            inner3[:, 1] += off2[:, 1]
            if _inner_valid(inner3) and _outer_contains(inner3, cx2, cy2, s2, ang2, tol=-1e-6):
                inner = inner3
                best = (s2, cx2, cy2, ang2)
    except Exception:
        pass

    s, cx, cy, ang = best
    outer = np.array([cx, cy, ang])
    return inner, outer, float(s)


# EVOLVE-BLOCK-END
