# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = 600.0


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0) * np.arange(6)
    return np.column_stack((cx + s * np.cos(th), cy + s * np.sin(th)))


def _outer_vertices(cx, cy, s, a_deg):
    return _hex_vertices(cx, cy, s, a_deg)


def _separation_gap(h1, h2):
    """Max over 12 edge normals of the gap between projected intervals.
    Positive => separated by that amount. Negative => overlap magnitude."""
    a1 = np.deg2rad(h1[2])
    a2 = np.deg2rad(h2[2])
    d = np.array([h2[0] - h1[0], h2[1] - h1[1]])
    best = -1e30
    for base in (a1, a2):
        for k in range(6):
            ang = base + np.deg2rad(60.0) * k
            n = np.array([np.cos(ang), np.sin(ang)])
            proj1 = np.dot(n, _hex_vertices(h1[0], h1[1], 1.0, h1[2]))
            proj2 = np.dot(n, _hex_vertices(h2[0], h2[1], 1.0, h2[2]))
            gap = min(proj2) - max(proj1)
            if gap > best:
                best = gap
    return best


def _outer_margin(h, outer):
    """Min signed distance from h's vertices to outer hexagon boundary (positive inside).
    Outer centered at (ox,oy) with side s, angle a. A point p is inside iff for every
    edge normal n (outward), dot(n, p - o) <= s*apothem. Margin = s*apothem - max over edges."""
    ox, oy, oa = outer[0], outer[1], outer[2]
    a = np.deg2rad(oa)
    apothem = outer[2] if False else (outer[2])  # placeholder, set below
    # apothem of outer hexagon = side * cos(30deg)
    s = outer[2]
    apothem = s * np.cos(np.pi / 6.0)
    verts = _hex_vertices(h[0], h[1], 1.0, h[2])
    worst = 1e30
    for k in range(6):
        ang = a + np.deg2rad(60.0) * k
        n = np.array([np.cos(ang), np.sin(ang)])
        proj = np.dot(verts, n) - np.dot(np.array([ox, oy]), n)
        # inside if proj <= apothem; margin = apothem - max(proj)
        m = apothem - np.max(proj)
        if m < worst:
            worst = m
    return worst


def _objective_and_valid(params, n=11):
    # params: [x1,y1,a1, ..., x11,y11,a11, s]  (outer centered at origin, angle 0)
    inner = params[:3 * n].reshape(n, 3)
    s = params[3 * n]
    # soft-min over all pairwise separations
    gaps = []
    for i in range(n):
        for j in range(i + 1, n):
            gaps.append(_separation_gap(inner[i], inner[j]))
    gaps = np.array(gaps)
    gmin = np.min(gaps)
    # soft-min over outer margins
    outer = np.array([0.0, 0.0, 0.0])
    margins = np.array([_outer_margin(inner[i], np.array([0.0, 0.0, s])) for i in range(n)])
    mmin = np.min(margins)
    # combined constraint: min(gmin, mmin) should be > 0
    cmin = min(gmin, mmin)
    # smooth penalty
    pen = 500.0 * np.exp(-20.0 * cmin)
    return s + pen, cmin, gmin, mmin


def _check_valid(inner, outer, s, eps=1e-6):
    if not np.all(np.isfinite(inner)) or not np.isfinite(s) or not np.isfinite(outer[0]):
        return False
    if s < np.sqrt(11.0) - 1e-9:
        return False
    # pairwise separation
    for i in range(11):
        for j in range(i + 1, 11):
            if _separation_gap(inner[i], inner[j]) <= eps:
                return False
    # inside outer
    for i in range(11):
        if _outer_margin(inner[i], outer) <= eps:
            return False
    return True


def hexagon_packing_11():
    n = 11
    # Seed: honeycomb-like arrangement
    def seed0():
        # ring of 6 around center, plus 5 more
        arr = [
            [0.0, 0.0, 0.0],
            [2.5, 0.0, 0.0],
            [1.25, 2.165, 0.0],
            [-1.25, 2.165, 0.0],
            [-2.5, 0.0, 0.0],
            [-1.25, -2.165, 0.0],
            [1.25, -2.165, 0.0],
            [5.0, 0.0, 0.0],
            [3.75, 2.165, 0.0],
            [3.75, -2.165, 0.0],
            [0.0, 4.33, 0.0],
        ]
        inner = np.array(arr, dtype=float)
        s = 7.0
        p = np.concatenate([inner.ravel(), [s]])
        return p

    best_params = seed0()
    best_val, best_cmin, best_gmin, best_mmin = _objective_and_valid(best_params, n)

    rng = np.random.default_rng(0)

    def refine(p0, maxiter=400):
        from scipy.optimize import minimize
        def f(p):
            return _objective_and_valid(p, n)[0]
        try:
            res = minimize(f, p0, method='Nelder-Mead',
                           options={'maxiter': maxiter, 'xatol': 1e-7, 'fatol': 1e-9})
            if res.success or res.fun < f(p0):
                return res.x
        except Exception:
            pass
        return p0

    # Multi-start: perturb seed
    start_list = [seed0()]
    for k in range(12):
        base = seed0()
        # random perturbations of positions and small angle jitter, scale s
        inner = base[:3 * n].reshape(n, 3).copy()
        inner[:, 0] += rng.normal(0, 0.4, n)
        inner[:, 1] += rng.normal(0, 0.4, n)
        inner[:, 2] += rng.normal(0, 5.0, n)
        s = base[3 * n] * rng.uniform(0.85, 1.05)
        p = np.concatenate([inner.ravel(), [s]])
        start_list.append(p)

    for idx, p0 in enumerate(start_list):
        if time.time() > _DEADLINE - 60:
            break
        p = refine(p0, maxiter=300)
        val, cmin, gmin, mmin = _objective_and_valid(p, n)
        if val < best_val:
            best_val, best_cmin, best_gmin, best_mmin = val, cmin, gmin, mmin
            best_params = p

    # Final polish on best with tighter tolerance
    if time.time() < _DEADLINE - 30:
        p = refine(best_params, maxiter=800)
        val, cmin, gmin, mmin = _objective_and_valid(p, n)
        if val < best_val:
            best_val, best_cmin, best_gmin, best_mmin = val, cmin, gmin, mmin
            best_params = p

    inner = best_params[:3 * n].reshape(n, 3)
    s = best_params[3 * n]
    outer = np.array([0.0, 0.0, 0.0])

    # Verify; if invalid, fall back to a known-good large container
    if not _check_valid(inner, outer, s):
        inner = seed0()[:3 * n].reshape(n, 3)
        s = 8.0
        outer = np.array([0.0, 0.0, 0.0])

    return inner, outer, float(s)


# EVOLVE-BLOCK-END
