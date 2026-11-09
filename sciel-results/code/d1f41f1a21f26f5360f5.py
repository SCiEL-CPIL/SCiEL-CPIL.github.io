# EVOLVE-BLOCK-START
import time
import numpy as np

_EPS = 1e-6
_TOL = 1e-8
_TIME_LIMIT = 600.0
_SAFE_MARGIN = 30.0


def _hex_vertices(c, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0) * np.arange(6)
    return np.column_stack([c[0] + s * np.cos(th), c[1] + s * np.sin(th)])


def _pair_gap(A, B):
    """A, B: (6,2) vertex arrays. Returns minimal edge-normal gap (>=0 disjoint)."""
    best = 1e18
    for M, P in ((A, B), (B, A)):
        E = np.roll(M, -1, axis=0) - M
        L = np.hypot(E[:, 0], E[:, 1])
        n = np.column_stack([E[:, 1], -E[:, 0]]) / L[:, None]
        d = P @ n.T
        for e in range(6):
            g = max(np.min(d[:, e]), -np.max(d[:, e]))
            if g < best:
                best = g
    return best


def _validate(inner, outer):
    """inner (11,3), outer (3,). Returns (ok, max_gap, min_cross)."""
    if not (np.all(np.isfinite(inner)) and np.all(np.isfinite(outer))):
        return False, 0.0, 0.0
    c = outer[:2]
    s = float(outer[2])
    if s < np.sqrt(11.0) - 1e-9:
        return False, 0.0, 0.0
    # inner edge lengths ~ 1
    for i in range(inner.shape[0]):
        v = _hex_vertices(inner[i, :2], inner[i, 2], 0.0)
        e = np.roll(v, -1, axis=0) - v
        if np.max(np.abs(np.hypot(e[:, 0], e[:, 1]) - 1.0)) > _TOL:
            return False, 0.0, 0.0
    # pairwise separation
    verts = [_hex_vertices(inner[i, :2], inner[i, 2], 0.0) for i in range(inner.shape[0])]
    max_gap = 1e18
    for i in range(inner.shape[0]):
        for j in range(i + 1, inner.shape[0]):
            g = _pair_gap(verts[i], verts[j])
            if g < max_gap:
                max_gap = g
    if max_gap <= _EPS:
        return False, max_gap, 0.0
    # containment
    ov = _hex_vertices(c, s, 0.0)
    min_cross = 1e18
    for i in range(inner.shape[0]):
        for p in verts[i]:
            p = p - c
            for k in range(6):
                p1 = ov[k] - c
                p2 = ov[(k + 1) % 6] - c
                cr = p2[0] * p[1] - p2[1] * p[0]
                if cr < min_cross:
                    min_cross = cr
    if min_cross < -_EPS:
        return False, max_gap, min_cross
    return True, max_gap, min_cross


def _enclosing_side(cents, a=0.0):
    """Minimal side of a regular hexagon (centered at origin, angle a) containing
    unit hexagons at centers `cents` (side 1)."""
    a = np.deg2rad(a)
    th = a + np.deg2rad(60.0) * np.arange(6)
    E = np.column_stack([np.cos(th), np.sin(th)])
    C = np.column_stack([np.sin(th), -np.cos(th)])  # outward edge normals
    D = cents @ C.T
    return float(np.max(D) + np.sqrt(3.0) / 2.0)


def _base_positions():
    h = np.sqrt(3.0) / 2.0
    return np.array([
        [0.0, 0.0],
        [1.0, h], [1.0, -h], [-1.0, h], [-1.0, -h],
        [2.0, 0.0], [0.0, 2 * h], [-2.0, 0.0], [0.0, -2 * h],
        [2.0, h], [2.0, -h],
    ])


def _refine(base, a, budget):
    """Local search: perturb centers, keep valid, minimize enclosing side.
    Returns (inner, outer, side)."""
    pos = base.copy()
    s_best = _enclosing_side(pos, a)
    best = pos.copy()
    rng = np.random.default_rng(12345)
    start = time.time()
    step = 0.08
    shrink = 0.92
    for _ in range(400000):
        if time.time() - start > budget:
            break
        if step < 1e-5:
            break
        cand = best + rng.normal(0.0, step, size=(11, 2))
        s = _enclosing_side(cand, a)
        if s >= s_best:
            step *= shrink
            continue
        # build full inner and validate
        inner = np.zeros((11, 3))
        inner[:, 0] = cand[:, 0]
        inner[:, 1] = cand[:, 1]
        inner[:, 2] = a
        outer = np.array([0.0, 0.0, s])
        ok, gap, _ = _validate(inner, outer)
        if ok:
            s_best = s
            best = cand
            step *= 1.02
        else:
            step *= shrink
    return best, s_best


def hexagon_packing_11():
    t0 = time.time()
    budget = _TIME_LIMIT - _SAFE_MARGIN - (time.time() - t0)
    if budget < 2.0:
        budget = 2.0

    base = _base_positions()
    a = 0.0
    s0 = _enclosing_side(base, a)

    # verify base packing is valid
    inner = np.zeros((11, 3))
    inner[:, 0] = base[:, 0]
    inner[:, 1] = base[:, 1]
    inner[:, 2] = a
    outer = np.array([0.0, 0.0, s0])
    ok, gap, cr = _validate(inner, outer)
    if not ok:
        # fallback: enlarge
        outer = np.array([0.0, 0.0, 8.0])
        s0 = 8.0
        return inner, outer, s0

    # refine with the bulk of the budget
    b, s_best = _refine(base, a, budget)

    # final validation
    inner = np.zeros((11, 3))
    inner[:, 0] = b[:, 0]
    inner[:, 1] = b[:, 1]
    inner[:, 2] = a
    outer = np.array([0.0, 0.0, s_best])
    ok, gap, cr = _validate(inner, outer)
    if not ok:
        # fall back to base
        inner[:, 0] = base[:, 0]
        inner[:, 1] = base[:, 1]
        outer = np.array([0.0, 0.0, s0])
        s_best = s0
    return inner, outer, float(s_best)


# EVOLVE-BLOCK-END
