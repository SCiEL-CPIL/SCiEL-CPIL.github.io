import math
import time
import numpy as np

N = 12
DEADLINE = 140.0  # wall-clock safety margin (evaluator kills at 150s)

# --- regular hexagon geometry -------------------------------------------
# A unit regular hexagon with circumradius 1, oriented so two edges are
# horizontal (top/bottom). Vertices: (cos(k*pi/3), sin(k*pi/3)), k=0..5.
BASE_VERTS = np.array([[math.cos(math.radians(k * 60.0)),
                        math.sin(math.radians(k * 60.0))] for k in range(6)])


def _rot(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def _hex_verts(cx, cy, side, theta):
    # side length = circumradius for a regular hexagon
    return cx + side * (BASE_VERTS @ _rot(theta).T)


def _separation(A, B):
    # A,B: (6,2) vertex arrays. Returns signed distance between the two
    # convex hexagons: >0 separated, ~0 touching, <0 overlap magnitude.
    # Uses the separating-axis test over all 12 edge normals.
    best = 0.0
    for i in range(6):
        for base, other in ((A, B), (B, A)):
            e = base[(i + 1) % 6] - base[i]
            L = math.hypot(e[0], e[1])
            if L < 1e-12:
                continue
            n = np.array([-e[1], e[0]]) / L
            pa = A @ n
            pb = B @ n
            lo = max(pa.min(), pb.min())
            hi = min(pa.max(), pb.max())
            best = max(best, lo - hi)
    return best


def _build(params, S):
    # params: (3N,) -> [x0,y0,th0, x1,y1,th1, ...]; S = outer side length
    cx, cy = 0.0, 0.0
    verts = []
    for i in range(N):
        x, y, th = params[3 * i], params[3 * i + 1], params[3 * i + 2]
        verts.append(_hex_verts(x, y, 1.0, th))
    # outer hexagon (centered at origin, side S)
    outer = _hex_verts(cx, cy, S, 0.0)
    return verts, outer


def _objective(params, S):
    verts, outer = _build(params, S)
    # inner-inner separation
    sep_ii = np.empty((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            s = _separation(verts[i], verts[j])
            sep_ii[i, j] = sep_ii[j, i] = s
    # inner-outer: require each inner vertex inside outer (>=0 margin)
    # distance of inner vertex to outer boundary along outer normals.
    sep_io = np.empty((N, 6))
    for i in range(N):
        vi = verts[i]
        for k in range(6):
            e = outer[(k + 1) % 6] - outer[k]
            L = math.hypot(e[0], e[1])
            n = np.array([-e[1], e[0]]) / L
            # point inside if vi.n <= outer_max_n  -> margin = outer_max_n - vi.n
            outer_max = outer @ n
            margin = outer_max.max() - (vi @ n).max()
            sep_io[i, k] = margin
    # margins: positive = feasible
    sep_ii_m = sep_ii + 0.05   # small inner gap
    sep_io_m = sep_io + 0.02
    # smooth penalty (softplus) of negative margins
    def soft(x):
        return np.logaddexp(0.0, x)  # = log(1+e^x); small for x<<0
    pen = 0.0
    pen += 50.0 * float(np.sum(soft(-sep_ii_m)))
    pen += 100.0 * float(np.sum(soft(-sep_io_m)))
    return S + pen


def _init_grid():
    # 3 rows x 4 columns of hexagons, honeycomb offset.
    # horizontal spacing 2 (touching), vertical spacing sqrt(3), offset 1.
    pts = []
    for r in range(3):
        for c in range(4):
            x = c * 2.0 + (1.0 if r % 2 else 0.0)
            y = r * math.sqrt(3.0)
            pts.append((x, y))
    # center the set at origin
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    mx, my = xs.mean(), ys.mean()
    pts = [(x - mx, y - my) for (x, y) in pts]
    params = np.zeros(3 * N)
    for i, (x, y) in enumerate(pts):
        params[3 * i] = x
        params[3 * i + 1] = y
        params[3 * i + 2] = 0.0
    return params


def construct():
    t0 = time.time()
    params0 = _init_grid()
    # estimate a good starting S from bounding box of the grid
    S0 = 8.0
    # refine S0 to just cover the initial (feasible) arrangement
    for trial in range(40):
        S = S0 + trial * 0.05
        _, outer = _build(params0, S)
        ok = True
        for i in range(N):
            vi = _hex_verts(params0[3 * i], params0[3 * i + 1], 1.0, params0[3 * i + 2])
            for k in range(6):
                e = outer[(k + 1) % 6] - outer[k]
                L = math.hypot(e[0], e[1])
                n = np.array([-e[1], e[0]]) / L
                if (vi @ n).max() > (outer @ n).max():
                    ok = False
                    break
            if not ok:
                break
        if ok:
            S0 = S
            break
    S0 = min(S0, 10.0)

    best_params = params0.copy()
    best_S = S0
    best_val = _objective(params0, S0)

    from scipy.optimize import minimize

    def obj(x):
        return _objective(x, best_S)

    # Optimize: first fix S, minimize penalty+... then reduce S.
    # Simpler: minimize over (params, S) jointly with S in a box.
    def joint(x):
        p = x[:3 * N]
        S = x[3 * N]
        return _objective(p, S)

    x0 = np.concatenate([params0, [S0]])
    lb = np.concatenate([np.full(3 * N, -20.0), [4.0]])
    ub = np.concatenate([np.full(3 * N, 20.0), [12.0]])
    bounds = list(zip(lb, ub))

    try:
        res = minimize(joint, x0, method='L-BFGS-B', bounds=bounds,
                       options={'maxiter': 400, 'maxfun': 2000,
                                'ftol': 1e-10, 'gtol': 1e-8})
        if time.time() - t0 > DEADLINE:
            raise TimeoutError()
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x[:3 * N].copy()
            best_S = float(res.x[3 * N])
    except Exception:
        pass

    # Second pass: with layout fixed at best_params, shrink S to the minimal
    # feasible value via bisection (robust, no gradient needed).
    if time.time() - t0 < DEADLINE - 5.0:
        def feasible(S):
            _, outer = _build(best_params, S)
            for i in range(N):
                vi = _hex_verts(best_params[3 * i], best_params[3 * i + 1], 1.0,
                                best_params[3 * i + 2])
                for k in range(6):
                    e = outer[(k + 1) % 6] - outer[k]
                    L = math.hypot(e[0], e[1])
                    n = np.array([-e[1], e[0]]) / L
                    if (vi @ n).max() > (outer @ n).max():
                        return False
            return True

        lo, hi = 4.0, best_S
        if feasible(best_S):
            for _ in range(60):
                if time.time() - t0 > DEADLINE - 3.0:
                    break
                mid = 0.5 * (lo + hi)
                if feasible(mid):
                    hi = mid
                else:
                    lo = mid
            best_S = hi

    # Build final inner rows
    inner = []
    for i in range(N):
        x = float(best_params[3 * i])
        y = float(best_params[3 * i + 1])
        ang = float(np.degrees(best_params[3 * i + 2]) % 360.0)
        inner.append([x, y, ang])
    return inner, (0.0, 0.0), float(best_S), 0.0
