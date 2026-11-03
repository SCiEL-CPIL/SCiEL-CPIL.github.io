"""Packing 12 unit hexagons in a minimal enclosing regular hexagon.
Continuous multi-start Nelder-Mead with a hard non-overlap penalty."""
import math
import time
import numpy as np

N = 12
_PI3 = math.pi / 3.0
_TOL = 1e-7
_BIG = 1e5

def _hex_verts(x, y, a):
    # flat-top regular hexagon, side 1, rotation a
    ca, sa = math.cos(a), math.sin(a)
    base = [(1.0, 0.0), (0.5, math.sqrt(3) / 2.0), (-0.5, math.sqrt(3) / 2.0),
            (-1.0, 0.0), (-0.5, -math.sqrt(3) / 2.0), (0.5, -math.sqrt(3) / 2.0)]
    return np.array([(x + ca * px - sa * py, y + sa * px + ca * py) for px, py in base])


def _separation(A, B):
    # largest positive separation between convex polys A,B over all SAT axes
    best = 0.0
    for V in (A, B):
        n = V.shape[0]
        for i in range(n):
            ax = V[(i + 1) % n, 0] - V[i, 0]
            ay = V[(i + 1) % n, 1] - V[i, 1]
            L = math.hypot(ax, ay)
            if L < 1e-12:
                continue
            ux, uy = ax / L, ay / L
            d = np.dot(V, (ux, uy))
            dA, dB = d.min(), d.max()
            e = dB - dA
            d = np.dot(B, (ux, uy))
            dA2, dB2 = d.min(), d.max()
            e2 = dB2 - dA2
            sep = max(dB - dA2, dB2 - dA)
            if sep > best:
                best = sep
    return best


def _max_overlap(V):
    # V: (N,6,2). Return largest penetration depth (<=0 means no overlap).
    mo = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            sep = _separation(V[i], V[j])
            if sep < -mo:
                mo = -sep
    return mo


def _outer_side(V, cx, cy, phi):
    # minimal side S of regular hexagon (center cx,cy, rot phi) containing all vertices
    ca, sa = math.cos(phi), math.sin(phi)
    base = [(1.0, 0.0), (0.5, math.sqrt(3) / 2.0), (-0.5, math.sqrt(3) / 2.0),
            (-1.0, 0.0), (-0.5, -math.sqrt(3) / 2.0), (0.5, -math.sqrt(3) / 2.0)]
    ux = np.array([ca * px - sa * py for px, py in base])
    uy = np.array([sa * px + ca * py for px, py in base])
    P = V.reshape(-1, 2)
    P = P - np.array([cx, cy])
    S = 0.0
    for k in range(6):
        t = P[:, 0] * ux[k] + P[:, 1] * uy[k]
        m = max(t.max(), -t.min())
        if m > S:
            S = m
    return S


def _obj(z):
    V = np.array([_hex_verts(z[3 * i], z[3 * i + 1], z[3 * i + 2]) for i in range(N)])
    S = _outer_side(V, z[0], z[1], z[2])
    mo = _max_overlap(V)
    return S + _BIG * max(0.0, mo)


def construct():
    t0 = time.time()
    deadline = t0 + 135.0
    rng = np.random.default_rng(12345)

    # initial cluster: hexagonal grid of 12 cells (side 2 spacing)
    grid = [(-1, -1), (0, -1), (1, -1), (-1, 0), (0, 0), (1, 0),
            (-1, 1), (0, 1), (1, 1), (-2, 0), (2, 0), (0, -2)]
    init = np.zeros(3 * N + 3)
    for i, (gx, gy) in enumerate(grid):
        init[3 * i] = 2.0 * (gx + 0.5 * gy)
        init[3 * i + 1] = math.sqrt(3) * gy
        init[3 * i + 2] = 0.0

    best = None
    best_val = None

    def consider(z):
        nonlocal best, best_val
        v = _obj(z)
        if best_val is None or v < best_val:
            best_val = v
            best = z.copy()

    # run 0: symmetric start
    consider(init)

    n_starts = 16
    for s in range(n_starts):
        if time.time() > deadline - 8.0:
            break
        z = init.copy()
        for i in range(N):
            z[3 * i] += rng.uniform(-0.4, 0.4)
            z[3 * i + 1] += rng.uniform(-0.4, 0.4)
            z[3 * i + 2] = rng.uniform(0.0, _PI3)
        z[0] = rng.uniform(-0.3, 0.3)
        z[1] = rng.uniform(-0.3, 0.3)
        z[2] = rng.uniform(0.0, _PI3)
        try:
            res = minimize(_obj, z, method='Nelder-Mead',
                           options={'maxiter': 350, 'xatol': 1e-4, 'fatol': 1e-5,
                                    'adaptive': True, 'initial_simplex_size': 0.1})
            consider(res.x)
        except Exception:
            pass

    if best is None:
        best = init
    # decode
    inner = [[float(best[3 * i]), float(best[3 * i + 1]),
              math.degrees(float(best[3 * i + 2]))] for i in range(N)]
    cx, cy = float(best[0]), float(best[1])
    phi_deg = math.degrees(float(best[2]))
    V = np.array([_hex_verts(best[3 * i], best[3 * i + 1], best[3 * i + 2]) for i in range(N)])
    S = _outer_side(V, cx, cy, best[2])
    return inner, (cx, cy), float(S), phi_deg
