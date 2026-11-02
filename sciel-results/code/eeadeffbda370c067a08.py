"""Fresh design: physics relaxation (repulsion + containment) for 12 unit hexagons."""
import time
import math
import random
import numpy as np

N = 12
_DEADLINE = time.time() + 145.0          # safety margin vs 150s kill
_EPS = 1e-9
_COS60 = 0.5
_SIN60 = math.sqrt(3.0) / 2.0

# Unit hexagon vertices (circumradius 1) in order, plus its 6 inward normals.
_VERT = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
_NORM = -_VERT


def _verts(c, ang):
    th = math.radians(ang)
    ct, st = math.cos(th), math.sin(th)
    R = np.array([[ct, -st], [st, ct]])
    return c + _VERT.T @ R


def _gap(A, B):
    """Chebyshev distance between two rotated unit hexagons (0 if overlapping)."""
    d = B - A
    m = 0.0
    for i in range(6):
        s = _NORM[i, 0] * d[0] + _NORM[i, 1] * d[1]
        if s > m:
            m = s
    return m


def _outer_side(P, oc):
    """Min side of an axis-aligned regular hexagon at oc enclosing points P."""
    if len(P) == 0:
        return 0.0
    rel = P - oc
    r = np.abs(rel)
    m = np.maximum(r[:, 0], _COS60 * r[:, 0] + _SIN60 * r[:, 1])
    return float(m.max())


def _evaluate(A, ang, oc):
    """Return (outer_side, max_overlap)."""
    V = np.vstack([_verts(A[i], ang[i]) for i in range(N)])
    side = _outer_side(V, oc)
    worst = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            g = _gap(A[i], A[j])
            if g < worst:
                worst = g
    return side, worst


def _repel(A):
    """Gradient contribution of pair repulsion to positions."""
    G = np.zeros_like(A)
    for i in range(N):
        for j in range(i + 1, N):
            d = A[i] - A[j]
            s = _NORM @ d
            k = int(np.argmin(s))
            f = math.exp(-s[k] * 2.5)
            nk = _NORM[k]
            G[i] += f * nk
            G[j] -= f * nk
    return G


def _relax(A0, ang0, oc0, budget):
    A = A0.copy(); ang = ang0.copy(); oc = np.array(oc0, dtype=float)
    best = None; best_side = float("inf"); best_ov = float("inf")
    t0 = time.time()
    T = 0.15
    while time.time() - t0 < budget:
        G = _repel(A)
        V = np.vstack([_verts(A[i], ang[i]) for i in range(N)])
        rel = V - oc
        r = np.abs(rel)
        m = np.maximum(r[:, 0], _COS60 * r[:, 0] + _SIN60 * r[:, 1])
        side = float(m.max())
        # center gradient: pull oc toward the point that sets the radius
        idx = int(np.argmax(m))
        p = rel[idx]
        sgn = np.sign(p)
        g = np.array([sgn[0], _COS60 * sgn[0] + _SIN60 * sgn[1]])
        g /= (np.linalg.norm(g) + _EPS)
        # angle gradient: rotate hexagon i to reduce its vertex that sets radius
        base = idx // 6
        a = _VERT[idx % 6]
        da = np.array([-a[1], a[0]])
        ang[base] += 0.02 * (a[0] * g[1] - a[1] * g[0])
        ang = np.mod(ang, math.tau)

        A += 0.05 * G - 0.004 * T * g
        oc += 0.004 * T * g

        ov = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                gg = _gap(A[i], A[j])
                if gg < ov:
                    ov = gg
        if ov < best_ov:
            best_ov = ov
            if side < best_side:
                best_side = side
                best = (A.copy(), ang.copy(), oc.copy())
        T *= 0.995
    return best, best_side, best_ov


def construct():
    random.seed(12345)
    np.random.seed(12345)
    A, ang, oc = None, None, None
    best_side = float("inf")
    best_ov = float("inf")
    t0 = time.time()
    while time.time() < _DEADLINE - 5.0:
        A0 = np.random.uniform(-1.5, 1.5, size=(N, 2))
        ang0 = np.array([random.choice([0.0, math.radians(30)]) for _ in range(N)])
        oc0 = (0.0, 0.0)
        b, s, o = _relax(A0, ang0, oc0, budget=min(4.0, max(0.5, _DEADLINE - 5.0 - time.time())))
        if b is not None:
            if o < best_ov:
                best_ov = o
                if s < best_side:
                    best_side = s
                    A, ang, oc = b
    if A is None:
        A = np.random.uniform(-1.5, 1.5, size=(N, 2)); ang = np.zeros(N); oc = np.array([0.0, 0.0])
    inner = [[float(A[i, 0]), float(A[i, 1]), float(math.degrees(ang[i]))] for i in range(N)]
    return inner, (float(oc[0]), float(oc[1])), float(best_side), 0.0
