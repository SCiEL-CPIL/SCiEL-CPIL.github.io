"""Sphere covering with 130 equal caps: optimize covering radius via smooth surrogate + Voronoi-vertex polish."""
import time
import numpy as np
from scipy.spatial import ConvexHull

N = 130
_DEADLINE = 150.0
_START = time.time()


def _remaining():
    return _DEADLINE - 10.0 - (time.time() - _START)  # 10s safety margin


def _normalize(P):
    n = np.linalg.norm(P, axis=1, keepdims=True)
    n[n < 1e-12] = 1.0
    return P / n


def _covering_radius(P):
    """Exact covering radius (radians) = max facet circumradius via ConvexHull."""
    try:
        hull = ConvexHull(P)
        # For each facet, compute the circumcenter (Voronoi vertex) and its angle.
        # The covering radius is the max over facets of the angle between the
        # facet's circumcenter direction and any of its vertices.
        best = 0.0
        for s in hull.simplices:
            tri = P[s]
            # circumcenter of the triangle on the unit sphere
            a, b, c = tri
            # solve for point equidistant: use cross-product formula
            # circumcenter of triangle on unit sphere
            x = np.cross(b - a, c - a)
            # normalize to get facet normal
            xn = np.linalg.norm(x)
            if xn < 1e-12:
                continue
            x = x / xn
            # circumcenter direction: a + alpha*(b-a) + beta*(c-a) equidistant
            # Simpler: circumcenter = (a+b+c) projected... not exact. Use linear solve.
            # Find v such that v.a = v.b = v.c (dot products equal) and ||v||=1
            # v = a + t1*(b-a) + t2*(c-a); enforce v.b = v.a and v.c = v.a
            ba = b - a
            ca = c - a
            # v.a = a.a + t1*(ba.a) + t2*(ca.a) = 1 + t1*ba@a + t2*ca@a
            # v.b = a.b + t1*(ba.b) + t2*(ca.b)
            # v.c = a.c + t1*(ba.c) + t2*(ca.c)
            # set v.b - v.a = 0, v.c - v.a = 0
            M = np.array([[ba @ ba, ca @ ba],
                          [ba @ ca, ca @ ca]])
            rhs = np.array([-(a @ ba), -(a @ ca)])
            try:
                t = np.linalg.solve(M, rhs)
            except np.linalg.LinAlgError:
                continue
            v = a + t[0]*ba + t[1]*ca
            nv = np.linalg.norm(v)
            if nv < 1e-12:
                continue
            v = v / nv
            # angle between v and a
            ang = np.arccos(np.clip(v @ a, -1.0, 1.0))
            if ang > best:
                best = ang
        return best
    except Exception:
        return np.pi


def _voronoi_vertices(P):
    """Return list of (vertex, facet_index) for each convex hull facet's circumcenter."""
    try:
        hull = ConvexHull(P)
    except Exception:
        return []
    out = []
    for fi, s in enumerate(hull.simplices):
        tri = P[s]
        a, b, c = tri
        ba = b - a
        ca = c - a
        M = np.array([[ba @ ba, ca @ ba],
                      [ba @ ca, ca @ ca]])
        rhs = np.array([-(a @ ba), -(a @ ca)])
        try:
            t = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            continue
        v = a + t[0]*ba + t[1]*ca
        nv = np.linalg.norm(v)
        if nv < 1e-12:
            continue
        v = v / nv
        out.append((v, fi))
    return out


def _soft_max_objective(P, probes, temp):
    """Soft-max of angular distances from each probe to nearest point. Returns (value, grad)."""
    # probes: (K,3) unit vectors; P: (N,3)
    # d[k] = min_i angle(probe_k, P_i)
    # We want soft-max over k of d[k], approximated by temp*log(sum(exp(d/temp)))
    # For gradient, use the argmin i* for each probe.
    K = probes.shape[0]
    Np = P.shape[0]
    # distances: cos of angle
    cosd = probes @ P.T  # (K, Np)
    cosd = np.clip(cosd, -1.0, 1.0)
    d = np.arccos(cosd)  # (K, Np)
    min_d = np.min(d, axis=1)  # (K,)
    argmin_i = np.argmin(d, axis=1)  # (K,)
    # soft-max
    m = np.max(min_d)
    w = np.exp((min_d - m) / temp)
    w = w / np.sum(w)
    val = m + temp * np.log(np.sum(w))
    # gradient of val w.r.t. P: sum_k w_k * grad_i* angle(probe_k, P_i*)
    # angle = arccos(p . v); d/dv arccos(p.v) = -p / sqrt(1-(p.v)^2)
    grad = np.zeros_like(P)
    for k in range(K):
        i = argmin_i[k]
        p = P[i]
        v = probes[k]
        dot = p @ v
        denom = np.sqrt(max(1.0 - dot*dot, 1e-12))
        grad[i] += w[k] * (-v / denom)
    return val, grad


def _project_tangent(P, G):
    """Project gradient G onto tangent spaces of the sphere at each P_i, keep on manifold."""
    return G - (G @ P.T) * P  # P is (N,3), G is (N,3): G_i - (G_i.P_i) P_i


def construct():
    t0 = time.time()

    # ---- Initial configuration: Fibonacci spiral ----
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z*z, 0.0))
    P = _normalize(np.column_stack([r*np.cos(phi), r*np.sin(phi), z]))

    best_P = P.copy()
    best_cr = _covering_radius(P)

    # Build probe set: Voronoi vertices of the current hull
    def build_probes(Pc):
        vv = _voronoi_vertices(Pc)
        if len(vv) >= 10:
            return np.array([v for v, _ in vv])
        # fallback: dense fibonacci grid
        M = 600
        j = np.arange(M) + 0.5
        zz = 1.0 - 2.0*j/M
        ph = np.pi*(1.0+np.sqrt(5.0))*j
        rr = np.sqrt(np.maximum(1.0-zz*zz, 0.0))
        return np.column_stack([rr*np.cos(ph), rr*np.sin(ph), zz])

    probes = build_probes(P)

    # ---- Gradient descent on the sphere with soft-max surrogate ----
    temp = 0.02  # softness
    lr = 0.05
    step = 0
    while _remaining() > 25.0:
        step += 1
        val, grad = _soft_max_objective(P, probes, temp)
        G = _project_tangent(P, grad)
        # adaptive step
        P_new = _normalize(P - lr * G)
        # exact check
        cr = _covering_radius(P_new)
        if cr < best_cr - 1e-12:
            best_cr = cr
            best_P = P_new.copy()
            P = P_new
            # rebuild probes periodically
            if step % 20 == 0 and _remaining() > 25.0:
                probes = build_probes(P)
            # decay learning rate slowly
            lr = max(0.005, lr * 0.999)
        else:
            # reject, shrink step
            lr = max(0.002, lr * 0.5)
            if lr <= 0.0021:
                # try smaller probe temp for sharper signal
                temp = max(0.005, temp * 0.5)
                lr = 0.01

    # ---- High-precision polish: local steepest descent with exact objective ----
    P = best_P.copy()
    best_cr = _covering_radius(P)
    lr = 0.02
    for _ in range(400):
        if _remaining() < 12.0:
            break
        probes = build_probes(P)
        val, grad = _soft_max_objective(P, probes, 0.01)
        G = _project_tangent(P, grad)
        P_new = _normalize(P - lr * G)
        cr = _covering_radius(P_new)
        if cr < best_cr - 1e-13:
            best_cr = cr
            best_P = P_new.copy()
            P = P_new
        else:
            lr = max(0.001, lr * 0.5)

    # Final normalization and validity guard
    P = _normalize(best_P)
    if not np.all(np.isfinite(P)):
        # regenerate fallback
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0*i/N
        phi = np.pi*(1.0+np.sqrt(5.0))*i
        r = np.sqrt(np.maximum(1.0-z*z, 0.0))
        P = _normalize(np.column_stack([r*np.cos(phi), r*np.sin(phi), z]))
    return P
