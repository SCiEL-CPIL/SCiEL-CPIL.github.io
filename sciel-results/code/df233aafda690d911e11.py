"""Heilbronn-11: multi-start local optimization in barycentric coords."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11
# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2.0])
H = np.sqrt(3) / 2.0
DEADLINE = 110.0  # safety margin under 120s / 150s kill


def _bary_to_cart(ab):
    """ab: (2N,) array of (a_i, b_i) barycentric coords -> (N,2) cartesian.
    c_i = 1 - a_i - b_i. point = a*V0 + b*V1 + c*V2."""
    a = ab[0::2]
    b = ab[1::2]
    c = 1.0 - a - b
    x = b * V1[0] + c * V2[0]
    y = b * V1[1] + c * V2[1]
    return np.column_stack([x, y])


def _all_areas(pts):
    """pts: (N,2). Return (C(N,3),) array of triangle areas."""
    n = pts.shape[0]
    idx_i = np.repeat(np.arange(n), n * (n - 1) // 2)
    # build all i<j<k
    ii, jj, kk = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ii.append(i); jj.append(j); kk.append(k)
    ii = np.array(ii); jj = np.array(jj); kk = np.array(kk)
    p_i = pts[ii]; p_j = pts[jj]; p_k = pts[kk]
    # area = 0.5 * |cross(p_j - p_i, p_k - p_i)|
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)


def _min_area(pts):
    return float(np.min(_all_areas(pts)))


def _softmin_objective(ab, tau):
    """Negative smooth min area + barrier. Returns (f, grad)."""
    pts = _bary_to_cart(ab)
    areas = _all_areas(pts)
    # softmin via log-sum-exp
    z = -areas / tau
    zmax = np.max(z)
    s = np.sum(np.exp(z - zmax))
    logsumexp = zmax + np.log(s)
    softmin = -tau * logsumexp
    # barrier for barycentric feasibility
    a = ab[0::2]; b = ab[1::2]; c = 1.0 - a - b
    mu = 1e-4
    barrier = 0.0
    # use softplus-like: penalize when coord < 0
    for coord in (a, b, c):
        neg = np.maximum(-coord, 0.0)
        barrier += mu * np.sum(neg * neg)
    f = -softmin + barrier
    # gradient via finite difference is too slow; use analytic for softmin
    # d(softmin)/d(areas_m) = exp(z_m)/sum(exp(z))  (with sign)
    # softmin = -tau*logsumexp(-areas/tau)
    # d softmin / d area_m = exp(-area_m/tau)/sum(exp(-area_k/tau))
    w = np.exp(z - zmax)
    w = w / np.sum(w)  # weights sum to 1
    # d softmin / d area_m = w_m
    # area_m = 0.5 * |cross|. For interior (non-degenerate) the sign is fixed.
    # We'll compute gradient w.r.t. ab via chain rule numerically per-point using
    # the signed cross (assume orientation consistent) — use finite diff fallback
    # for robustness is too slow; instead compute analytic via signed cross.
    # Recompute with signed cross to get linear areas (smooth).
    n = pts.shape[0]
    ii, jj, kk = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ii.append(i); jj.append(j); kk.append(k)
    ii = np.array(ii); jj = np.array(jj); kk = np.array(kk)
    p_i = pts[ii]; p_j = pts[jj]; p_k = pts[kk]
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    # use signed cross; area = 0.5*|cross|. For smoothness use 0.5*cross where cross>0
    # To keep it simple and smooth, use 0.5*cross (signed). softmin of signed can be
    # problematic if some are negative. Use abs via smooth: area = 0.5*(cross + |cross|)
    # gradient of |cross| = sign(cross). We'll use that.
    sign = np.sign(cross)
    # d area_m / d cross_m = 0.5 * sign(cross_m)
    # d cross_m / d ab: cross = (x_j-x_i)(y_k-y_i) - (y_j-y_i)(x_k-x_i)
    # This is linear in each point coordinate. Build Jacobian.
    grad_ab = np.zeros(2 * n)
    for m in range(len(ii)):
        wm = w[m] * 0.5 * sign[m]
        i, j, k = ii[m], jj[m], kk[m]
        xi, yi = pts[i]; xj, yj = pts[j]; xk, yk = pts[k]
        # d cross / d x_i = -(y_k - y_i); d cross / d y_i = (x_k - x_i)
        # d cross / d x_j = (y_k - y_i);  d cross / d y_j = -(x_k - x_i)
        # d cross / d x_k = -(y_j - y_i); d cross / d y_k = (x_j - x_i)
        dxi = -(yk - yi); dyi = (xk - xi)
        dxj = (yk - yi);  dyj = -(xk - xi)
        dxk = -(yj - yi); dyk = (xj - xi)
        # map cartesian grad to barycentric ab grad
        # point_p = b_p*V1 + c_p*V2, c=1-a-b
        # x = b*1 + c*0.5 = b + (1-a-b)*0.5 = 0.5 + 0.5b - 0.5a
        #   dx/da = -0.5, dx/db = 0.5
        # y = b*0 + c*H = (1-a-b)*H
        #   dy/da = -H, dy/db = -H
        def to_ab(gx, gy):
            ga = gx * (-0.5) + gy * (-H)
            gb = gx * (0.5) + gy * (-H)
            return ga, gb
        ga, gb = to_ab(wm * dxi, wm * dyi); grad_ab[2 * i] += ga; grad_ab[2 * i + 1] += gb
        ga, gb = to_ab(wm * dxj, wm * dyj); grad_ab[2 * j] += ga; grad_ab[2 * j + 1] += gb
        ga, gb = to_ab(wm * dxk, wm * dyk); grad_ab[2 * k] += ga; grad_ab[2 * k + 1] += gb
    # barrier gradient
    a = ab[0::2]; b = ab[1::2]; c = 1.0 - a - b
    for idx, coord in ((0, a), (1, b)):
        neg = np.where(coord < 0, -2 * mu * coord, 0.0)
        grad_ab[idx::2] += neg
    # c = 1-a-b: dc/da=-1, dc/db=-1
    neg_c = np.where(c < 0, -2 * mu * c, 0.0)
    grad_ab[0::2] += -neg_c
    grad_ab[1::2] += -neg_c
    # f = -softmin + barrier, so grad f = -grad softmin + grad barrier
    grad_ab = -grad_ab
    return f, grad_ab


def _optimize_from(ab0, tau, time_left):
    """Run L-BFGS-B from ab0. Returns best ab, best val."""
    best_ab = ab0.copy()
    best_val = _min_area(_bary_to_cart(ab0))

    def obj(ab):
        f, g = _softmin_objective(ab, tau)
        return f, g

    try:
        res = minimize(obj, ab0, jac=True, method='L-BFGS-B',
                       options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-7})
        if res.x is not None:
            v = _min_area(_bary_to_cart(res.x))
            if v > best_val:
                best_val = v
                best_ab = res.x
    except Exception:
        pass
    return best_ab, best_val


def _grid_starts():
    """Structured triangular-grid starts in barycentric space."""
    starts = []
    # triangular lattice of MxM in (a,b) with a>=0,b>=0,a+b<=1
    for M in (4, 5):
        for ia in range(M + 1):
            for ib in range(M + 1 - ia):
                a = ia / M
                b = ib / M
                # replicate to N points: place point i at a rotated/shifted version
                ab = np.zeros(2 * N)
                for i in range(N):
                    # spread points around the lattice cell using i-dependent offset
                    ang = 2 * np.pi * i / N
                    r = 0.02
                    ai = np.clip(a + r * np.cos(ang), 0.0, 1.0)
                    bi = np.clip(b + r * np.sin(ang), 0.0, 1.0)
                    # ensure a+b<=1
                    s = ai + bi
                    if s > 1.0:
                        ai *= (1.0 - 1e-6) / s
                        bi *= (1.0 - 1e-6) / s
                    ab[2 * i] = ai
                    ab[2 * i + 1] = bi
                starts.append(ab)
    return starts


def _random_starts(rng, count):
    starts = []
    for _ in range(count):
        ab = np.zeros(2 * N)
        for i in range(N):
            a = rng.random()
            b = rng.random() * (1.0 - a)
            ab[2 * i] = a
            ab[2 * i + 1] = b
        starts.append(ab)
    return starts


def _symmetric_start():
    """A symmetric 11-point layout: center + 2 rings (1 + 6 + ... ) ."""
    # Use a hexagonal-ish arrangement in barycentric coords.
    ab = np.zeros(2 * N)
    # center
    ab[0] = 1.0 / 3.0
    ab[1] = 1.0 / 3.0
    # ring of 6 at radius r in barycentric plane
    r1 = 0.18
    for i in range(6):
        ang = 2 * np.pi * i / 6
        a = 1.0 / 3.0 + r1 * np.cos(ang)
        b = 1.0 / 3.0 + r1 * np.sin(ang)
        s = a + b
        if s > 1.0:
            a *= (1 - 1e-6) / s
            b *= (1 - 1e-6) / s
        ab[2 * (1 + i)] = a
        ab[2 * (1 + i) + 1] = b
    # ring of 4 at corners-ish
    r2 = 0.30
    for i in range(4):
        ang = 2 * np.pi * i / 4 + np.pi / 4
        a = 1.0 / 3.0 + r2 * np.cos(ang)
        b = 1.0 / 3.0 + r2 * np.sin(ang)
        s = a + b
        if s > 1.0:
            a *= (1 - 1e-6) / s
            b *= (1 - 1e-6) / s
        ab[2 * (7 + i)] = a
        ab[2 * (7 + i) + 1] = b
    return ab


def construct():
    t0 = time.time()
    rng = np.random.RandomState(12345)

    best_ab = None
    best_val = -1.0

    # Build candidate starts
    starts = []
    starts.append(_symmetric_start())
    starts.extend(_grid_starts())
    starts.extend(_random_starts(rng, 12))

    # anneal tau from coarse to fine
    taus = [0.05, 0.02, 0.008, 0.003, 0.001]

    # warm start: optimize symmetric first with coarse tau
    current = _symmetric_start()
    for tau in taus:
        if time.time() - t0 > DEADLINE:
            break
        current, v = _optimize_from(current, tau, DEADLINE - (time.time() - t0))
        if v > best_val:
            best_val = v
            best_ab = current

    # multi-start from grid/random with a mid tau
    for ab0 in starts:
        if time.time() - t0 > DEADLINE:
            break
        for tau in [0.02, 0.005]:
            if time.time() - t0 > DEADLINE:
                break
            ab, v = _optimize_from(ab0, tau, DEADLINE - (time.time() - t0))
            if v > best_val:
                best_val = v
                best_ab = ab

    if best_ab is None:
        best_ab = _symmetric_start()

    pts = _bary_to_cart(best_ab)
    # clamp into triangle for safety
    eps = 1e-9
    for i in range(N):
        x, y = pts[i]
        # constraints: y>=0, y<=H*2*x? Actually triangle (0,0),(1,0),(0.5,H)
        # y >= 0
        # y <= H * (1 - |2x - 1|)  (the two slanted edges)
        # left edge from (0,0) to (0.5,H): y = 2H x  -> y <= 2H x
        # right edge from (1,0) to (0.5,H): y = 2H (1 - x) -> y <= 2H(1-x)
        y = max(eps, min(y, 2 * H * x, 2 * H * (1 - x)))
        x = max(eps, min(1 - eps, x))
        pts[i] = [x, y]
    return np.array(pts)
