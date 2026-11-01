import math
import time
import numpy as np

N = 11
T0 = time.time()
DEADLINE = T0 + 110.0
EPS = 1e-9

# --- axis-aligned unit hexagon geometry -----------------------------------
# vertices (pointy-top), side 1
_VX = np.array([0.0, 1.0, 1.0, 0.0, -1.0, -1.0])
_VY = np.array([math.sqrt(3)/2, math.sqrt(3)/2, -math.sqrt(3)/2, -math.sqrt(3)/2,
                -math.sqrt(3)/2, math.sqrt(3)/2])
# six outward face normals
_NX = np.array([0.0, 1.0, math.sqrt(3)/2, 0.0, -1.0, -math.sqrt(3)/2])
_NY = np.array([1.0, 0.0, -0.5, -1.0, 0.0, 0.5])
# support of a unit hexagon in each face direction (dot of normal with vertices)
_H = np.array([math.sqrt(3)/2, 1.0, math.sqrt(3)/2, math.sqrt(3)/2, 1.0, math.sqrt(3)/2])
_HH = _H + _H  # sum of supports for two unit hexagons

# enclosing-hexagon direction set (30 deg spacing)
_A = np.array([math.radians(k) for k in range(0, 180, 30)])
_AX = np.cos(_A)
_AY = np.sin(_A)


def enclosing_side(x):
    """Minimum side length of an axis-aligned regular hexagon containing the set."""
    px = _AX * x[:, 0] + _AY * x[:, 1]
    span = px.max(axis=1) - px.min(axis=1)
    return 2.0 * span / math.sqrt(3.0)


def objective(z):
    """Smooth objective: enclosing side + overlap penalty. Lower is better."""
    x = z.reshape(N, 2)
    cx = x[:, 0]
    cy = x[:, 1]
    d = cx[:, None] - cx[None, :]
    e = cy[:, None] - cy[None, :]
    gap = np.empty((N, N, 6))
    for k in range(6):
        gap[:, :, k] = _NX[k] * d + _NY[k] * e - _HH[k]
    over = np.maximum(gap, 0.0)
    s = enclosing_side(x)
    return s + 0.5 * np.sum(over * over)


def _feasible(x):
    cx = x[:, 0]
    cy = x[:, 1]
    d = cx[:, None] - cx[None, :]
    e = cy[:, None] - cy[None, :]
    for k in range(6):
        if np.any(_NX[k] * d + _NY[k] * e - _HH[k] > EPS):
            return False
    return True


def _recenter(x):
    cx = x[:, 0]
    cy = x[:, 1]
    best = None
    bx = by = 0.0
    for sx in (-0.5, 0.0, 0.5):
        for sy in (-0.5, 0.0, 0.5):
            s = enclosing_side(x + np.array([sx, sy]))
            if best is None or s < best:
                best = s
                bx, by = sx, sy
    return best, bx, by


def _de(z, pop, gen, scale, cr):
    rng = np.random.default_rng()
    best = z
    bestv = objective(z)
    for g in range(gen):
        if time.time() > DEADLINE:
            break
        for i in range(pop):
            a, b, c = rng.choice(pop, size=3, replace=False)
            v = z[a] + scale * (z[b] - z[c])
            for j in range(N * 2):
                if rng.random() < cr:
                    v[j] = z[i][j]
            vv = objective(v)
            if vv < bestv:
                best = v.copy()
                bestv = vv
            if vv < objective(z[i]):
                z[i] = v
    return best, bestv


def _nelder(z, max_iter):
    x = z.copy()
    f = [objective(x)]
    for _ in range(N * 2):
        xp = x.copy()
        xp[_] += 0.05
        f.append(objective(xp))
        x = xp
    alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
    for it in range(max_iter):
        if time.time() > DEADLINE:
            break
        order = np.argsort(f)
        xw, fw = x[order[-1]], f[order[-1]]
        xc = np.mean(x[order[:-(N * 2 + 1) + 1] if False else order[:-1]], axis=0)
        xr = xc + alpha * (xc - xw)
        fr = objective(xr)
        if fr < f[order[0]]:
            xe = xc + gamma * (xc - xw)
            fe = objective(xe)
            if fe < fr:
                x[order[-1]], f[order[-1]] = xe, fe
            else:
                x[order[-1]], f[order[-1]] = xr, fr
        elif fr < f[order[-2]]:
            x[order[-1]], f[order[-1]] = xr, fr
        else:
            for j in range(len(x)):
                if j == order[-1]:
                    continue
                x[j] = (x[j] + x[order[-1]]) * 0.5
                f[j] = objective(x[j])
    order = np.argsort(f)
    return x[order[0]], f[order[0]]


def construct():
    # initial spread along a line (matches seed layout, guaranteed disjoint)
    rng = np.random.default_rng(12345)
    z0 = np.zeros(N * 2)
    for i in range(N):
        z0[2 * i] = 4.0 * (i - N // 2)
        z0[2 * i + 1] = 0.0
    z = z0.copy()

    # global differential evolution
    pop = 40
    Z = np.empty((pop, N * 2))
    Z[0] = z0
    for i in range(1, pop):
        Z[i] = z0 + rng.normal(0.0, 6.0, N * 2)
    best, bestv = _de(Z, pop, 25, scale=0.8, cr=0.9)
    z = best

    # local polish
    for _ in range(6):
        if time.time() > DEADLINE:
            break
        z, _ = _nelder(z, max_iter=400)

    x = z.reshape(N, 2)
    if not _feasible(x):
        x = z0.reshape(N, 2)

    s, bx, by = _recenter(x)
    cx = np.mean(x[:, 0]) + bx
    cy = np.mean(x[:, 1]) + by
    inner = [[float(x[i, 0] - cx), float(x[i, 1] - cy), 0.0] for i in range(N)]
    return inner, (float(cx), float(cy)), float(s), 0.0
