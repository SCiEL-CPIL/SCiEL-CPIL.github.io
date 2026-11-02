import time
import numpy as np
import scipy.optimize as opt

N = 12
START = time.time()
DEADLINE = 150.0

# ---- Geometry helpers (unit hexagon, flat-top, circumradius = 1) ----
A = np.array([0.0, 1.0, np.sqrt(3)/2, 0.0, -np.sqrt(3)/2, -1.0])  # unit vertices
C = np.array([0.0, 0.5, 0.5, 0.0, -0.5, -0.5])                   # outward normals
NORM = np.array([1.0, np.sqrt(3)/2, np.sqrt(3)/2, 1.0, np.sqrt(3)/2, np.sqrt(3)/2])
D = NORM / NORM.sum()  # equal weight per face
ROT = np.array([[np.cos(np.pi/3), -np.sin(np.pi/3)],
                [np.sin(np.pi/3),  np.cos(np.pi/3)]])
D6 = np.array([0.0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])
EPS = 1e-9

def _rot(p, ang):
    c = np.cos(ang); s = np.sin(ang)
    return p * c + np.cross([0, 0, 1], p) * s

def _hex_dist(p, q):
    d = p - q
    return max(np.abs(np.dot(d, C)), np.abs(np.dot(d, C @ ROT)), np.abs(np.dot(d, C @ (ROT @ ROT))))

def _dist6(p, q):
    d = p - q
    return np.array([np.dot(d, C), np.dot(d, C @ ROT), np.dot(d, C @ (ROT @ ROT))])

def _penalty(pos, ang):
    # overlap (pairwise) + containment (vs outer hexagon centered at origin, side R=1)
    pen = 0.0
    for i in range(N):
        pi = pos[i]
        for j in range(i + 1, N):
            pen += max(0.0, 2.0 - _hex_dist(pi, pos[j]))
        for k in range(6):
            pen += max(0.0, 1.0 - np.dot(_rot(pi, ang[i] + D6[k]), C))
    return pen

def _obj(z):
    pos = z[:2*N].reshape(N, 2)
    ang = z[2*N:]
    return _penalty(pos, ang)

def _try(seed):
    pos = seed[:2*N].reshape(N, 2).copy()
    ang = seed[2*N:].copy()
    p0 = np.concatenate([pos.ravel(), ang])
    p0 = np.clip(p0, -1.6, 1.6)
    res = opt.minimize(_obj, p0, method='SLSQP',
                       bounds=[(-1.6, 1.6)] * (3 * N),
                       options={'maxiter': 200, 'ftol': 1e-6, 'disp': False})
    if not res.success:
        return None
    pos = res.x[:2*N].reshape(N, 2)
    ang = res.x[2*N:]
    pen = _penalty(pos, ang)
    if pen > 1e-7:
        return None
    # compute minimal containing hexagon side R (origin-centered)
    R = 0.0
    for i in range(N):
        for k in range(6):
            R = max(R, np.dot(_rot(pos[i], ang[i] + D6[k]), C))
    return pos, ang, R

def _build(pos, ang, R):
    inner = []
    for i in range(N):
        inner.append([float(pos[i][0]), float(pos[i][1]), float(np.degrees(ang[i]))])
    return inner, (0.0, 0.0), float(R), 0.0

def _valid(inner, center, R, ang):
    ox, oy = center
    P = np.array([[p[0] - ox, p[1] - oy] for p in inner])
    Aa = np.array([[np.cos(np.radians(a)), -np.sin(np.radians(a))] for a in ang])
    # outer hexagon: vertices at R * A, normals R * C (origin-centered)
    for i in range(N):
        v = A @ (Aa[i] @ P[i])
        for k in range(6):
            if np.dot(v, C[k]) > R * NORM[k] + 1e-6:
                return False
    for i in range(N):
        for j in range(i + 1, N):
            if _hex_dist(P[i], P[j]) < 2.0 - 1e-6:
                return False
    return True

def construct():
    best = None
    rng = np.random.default_rng(7)
    # initial hexagonal lattice packing (12 around center), angles aligned
    base_pos = []
    for k in range(12):
        a = np.radians(60 * k)
        base_pos.append([2.0 * np.cos(a), 2.0 * np.sin(a)])
    base_pos.append([0.0, 0.0])
    base_pos = base_pos[:12]
    base_ang = np.zeros(12)
    base_seed = np.concatenate([np.array(base_pos).ravel(), base_ang])
    try:
        r = _try(base_seed)
        if r is not None:
            best = r
    except Exception:
        pass

    # random restarts
    while time.time() < DEADLINE - 25.0:
        ang = np.array([np.radians(rng.integers(0, 6)) * (np.pi / 3) + rng.uniform(-0.3, 0.3)
                        for _ in range(12)])
        pos = np.array([[rng.uniform(-1.4, 1.4), rng.uniform(-1.4, 1.4)] for _ in range(12)])
        seed = np.concatenate([pos.ravel(), ang])
        try:
            r = _try(seed)
        except Exception:
            continue
        if r is not None and (best is None or r[2] < best[2]):
            best = r

    if best is None:
        # fallback: seed program (valid, side 40)
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    pos, ang, R = best
    inner, center, side, outer_ang = _build(pos, ang, R)
    if not _valid(inner, center, side, ang):
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    return inner, center, side, outer_ang
