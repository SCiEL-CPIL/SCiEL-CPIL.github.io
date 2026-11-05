"""Fresh design: honeycomb packing of 12 unit hexagons + exact minimal enclosing hexagon via LP."""
import math
import time
import numpy as np

N = 12
_T0 = time.time()
_DEADLINE = _T0 + 135.0          # safety margin before the 150 s kill
_EPS = 1e-3                      # robustness gap so hexagons are strictly disjoint
_R = 1.0                         # circumradius of a unit (side=1) hexagon
_SQRT3 = math.sqrt(3.0)


def _hex_vertices(c, ang_deg):
    """Vertices of a unit (side=1) regular hexagon centered at c=(x,y), rotated by ang_deg."""
    a = math.radians(ang_deg)
    ca, sa = math.cos(a), math.sin(a)
    out = []
    for k in range(6):
        t = a + math.pi * k / 3.0
        x = _R * math.cos(t)
        y = _R * math.sin(t)
        out.append((c[0] + ca * x - sa * y, c[1] + sa * x + ca * y))
    return np.array(out)


def _chebyshev_side(verts, ang_deg):
    """Minimal side of a regular hexagon (center free, fixed orientation ang_deg) enclosing verts.
    Returns (side, center) or (inf, None)."""
    a = math.radians(ang_deg)
    # face normals of the outer hexagon (pointing outward)
    nrm = np.array([[math.cos(a + math.pi * k / 3.0) for k in range(6)]])
    nrm = nrm.T  # (6,2)
    proj = verts @ nrm.T  # (M,6)  proj[i,k] = v_i . n_k
    pmin = proj.min(axis=0)
    pmax = proj.max(axis=0)
    # side s with center c:  pmin_k + c.n_k <= -s  and  pmax_k + c.n_k <= +s
    #  =>  -pmin_k - s <= c.n_k <= pmax_k + s  for all k
    A_ub = np.vstack([nrm, -nrm])            # (12,2)
    b_ub = np.concatenate([-pmin, pmax])     # (12,)
    # objective: minimize s  ->  minimize [0,0,1] . [c_x, c_y, s]
    c_obj = np.array([0.0, 0.0, 1.0])
    # s >= 0
    A_eq = np.array([[0.0, 0.0, -1.0]])
    b_eq = np.array([0.0])
    try:
        from scipy.optimize import linprog
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                      bounds=[(None, None), (None, None), (0.0, None)],
                      method="highs")
        if not res.success:
            return math.inf, None
        side = float(res.x[2])
        center = (float(res.x[0]), float(res.x[1]))
        return side, center
    except Exception:
        return math.inf, None


def _best_for_spacing(spacing):
    """Honeycomb layout for given center-to-center spacing; return best (side, inner, center, ang)."""
    # Row layout: 4 rows, columns 3,4,4,3 -> 12 hexagons, 60-degree staggered.
    rows = [
        ([-1.5, -0.5, 0.5, 1.5], 0.0),      # 4 cols, y = 0
        ([-1.0, 0.0, 1.0], _SQRT3 * spacing / 2.0),   # 3 cols, y = +h
        ([-1.5, -0.5, 0.5, 1.5], _SQRT3 * spacing),   # 4 cols, y = +2h
        ([-1.0, 0.0, 1.0], 1.5 * _SQRT3 * spacing),   # 3 cols, y = +3h
    ]
    inner = []
    for cols, y in rows:
        for cx in cols:
            inner.append([cx * spacing, y, 0.0])
    verts = np.vstack([_hex_vertices((p[0], p[1]), p[2]) for p in inner])

    best = (math.inf, None, None, None)
    for ang in range(0, 180, 3):          # coarse orientation scan
        side, center = _chebyshev_side(verts, ang)
        if side < best[0]:
            best = (side, inner, center, ang)
    # refine around the best coarse angle
    if best[3] is not None:
        for da in np.linspace(-3.0, 3.0, 13):
            ang = best[3] + da
            side, center = _chebyshev_side(verts, ang)
            if side < best[0]:
                best = (side, inner, center, float(ang))
    return best


def construct():
    best = (math.inf, None, None, None)
    # 1) coarse spacing search
    for spacing in np.arange(2.0 + _EPS, 2.6, 0.05):
        if time.time() > _DEADLINE:
            break
        cand = _best_for_spacing(float(spacing))
        if cand[0] < best[0]:
            best = cand
    # 2) bisection polish on spacing near the best (toward touching limit)
    if best[0] < math.inf:
        lo, hi = 2.0 + _EPS, 2.6
        # find a feasible hi that is >= best spacing
        best_sp = None
        for spacing in np.arange(2.0 + _EPS, 2.6, 0.05):
            if _best_for_spacing(float(spacing))[0] <= best[0] + 1e-9:
                best_sp = float(spacing)
        if best_sp is not None:
            hi = best_sp
            for _ in range(6):
                if time.time() > _DEADLINE:
                    break
                mid = 0.5 * (lo + hi)
                cand = _best_for_spacing(mid)
                if cand[0] < best[0]:
                    best = cand
                    hi = mid
                else:
                    lo = mid
    if best[1] is None:
        # absolute fallback (should not happen)
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    return best[1], best[2], best[0], best[3]
