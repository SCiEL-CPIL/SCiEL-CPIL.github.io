"""SCiEL-evolved program: high-symmetry candidate seeding + SLSQP polish."""
import time
import numpy as np
from scipy.optimize import minimize

N = 14
D = 3
# 91 pairwise indices for n=14
_IDX = np.array([(i, j) for i in range(N) for j in range(i + 1, N)], dtype=int)

DEADLINE = time.time() + 120.0 - 15.0   # safety margin


def _score(pts):
    d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
    iu = np.triu_indices(N, 1)
    dd = d[iu]
    lo = dd.min()
    hi = dd.max()
    if lo <= 1e-12:
        return 1e30
    return (hi / lo) ** 2


def _objective(x, tau=5000.0):
    """Smooth log-domain objective: minimize (1/tau) logsumexp(log d^2)."""
    pts = x.reshape(N, D)
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = (diff ** 2).sum(-1)[_IDX]
    d2 = np.maximum(d2, 1e-12)
    return np.log(np.sum(np.exp(np.log(d2) / tau)) / len(d2))


def _cube_octahedron(alpha):
    """6 cube vertices + 6 octahedron vertices, octahedron scaled by alpha."""
    cube = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
                     [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]], dtype=float)
    octa = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
                     [0, 0, 1], [0, 0, -1]], dtype=float)
    return np.vstack([cube, alpha * octa])


def _fibonacci_sphere(n):
    pts = np.zeros((n, 3))
    phi = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        z = 1.0 - (i / (n - 1)) * 2.0
        r = np.sqrt(max(0.0, 1.0 - z * z))
        th = phi * i
        pts[i] = [np.cos(th) * r, np.sin(th) * r, z]
    return pts


def _rand_pts():
    return np.random.randn(N, 3)


def _candidates():
    cands = []
    # 1) cube + octahedron composites with varying alpha
    for alpha in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.6, 2.0]:
        cands.append(_cube_octahedron(alpha))
    # 2) spherical fibonacci
    cands.append(_fibonacci_sphere(N))
    # 3) random restarts
    for _ in range(30):
        cands.append(_rand_pts())
    return cands


def construct():
    t0 = time.time()
    rng = np.random.RandomState(12345)
    np.random.seed(12345)
    cands = _candidates()

    best_pts = None
    best_val = np.inf
    for c in cands:
        v = _score(c)
        if v < best_val:
            best_val = v
            best_pts = c.copy()
        if time.time() > DEADLINE:
            return best_pts

    # Polish top candidates with SLSQP
    # rank candidates by score, take a few
    ranked = sorted(cands, key=lambda c: _score(c))
    top = ranked[:4]
    if time.time() > DEADLINE:
        return best_pts

    for c in top:
        if time.time() > DEADLINE:
            return best_pts
        x0 = c.reshape(-1)
        try:
            res = minimize(_objective, x0, method='SLSQP',
                           options={'maxiter': 500, 'ftol': 1e-12, 'disp': False})
            if res.success or res.fun < _objective(x0, 5000.0):
                cand_pts = res.x.reshape(N, D)
                v = _score(cand_pts)
                if v < best_val:
                    best_val = v
                    best_pts = cand_pts.copy()
        except Exception:
            pass

    return best_pts
