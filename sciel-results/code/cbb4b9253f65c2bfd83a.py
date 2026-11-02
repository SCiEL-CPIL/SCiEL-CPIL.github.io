"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14
D = 3

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


def _pairwise(x):
    """Return (14,14) matrix of squared pairwise distances."""
    diff = x[:, None, :] - x[None, :, :]
    d2 = np.einsum('ijk,ijk->ij', diff, diff)
    return d2


def _score(x):
    d2 = _pairwise(x)
    iu = np.triu_indices(N, 1)
    v = d2[iu]
    return (v.max() / v.min()) ** 2


def _smooth_ratio(x, beta):
    """Smooth surrogate for (max/min)^2 using log-sum-exp."""
    d2 = _pairwise(x)
    iu = np.triu_indices(N, 1)
    v = d2[iu]
    logv = np.log(v + 1e-12)
    L = 0.5 * (logv.max() - logv.min())
    logmax = np.log(np.exp(beta * logv).sum()) / beta
    logmin = -np.log(np.exp(-beta * logv).sum()) / beta
    return np.exp(2 * (logmax - logmin))


def _objective(x, beta):
    return _smooth_ratio(x, beta)


def _icosahedron():
    phi = (1 + np.sqrt(5.0)) / 2.0
    pts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
    ], dtype=float)
    n = np.linalg.norm(pts, axis=1)
    return pts / n[:, None]


def _make_starts():
    starts = []
    # Start 0: icosahedron + two points along the 5-fold axis (high symmetry).
    ic = _icosahedron()
    s0 = np.vstack([ic, np.array([[0.0, 0.0, 1.0]]), np.array([[0.0, 0.0, -1.0]])])
    starts.append(s0)
    # Start 1: icosahedron + two points at opposite face-centers.
    f = ic[0] + ic[1] + ic[4]
    f = f / np.linalg.norm(f)
    s1 = np.vstack([ic, np.array([f]), np.array([-f])])
    starts.append(s1)
    rng = np.random.default_rng(0)
    for _ in range(8):
        pts = rng.normal(size=(N, D))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        starts.append(pts)
    return starts


def construct():
    deadline = time.time() + 110.0
    starts = _make_starts()
    best_x = None
    best_s = np.inf
    for s in starts:
        if time.time() > deadline:
            break
        res = minimize(_objective, s, args=(4.0,), method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8})
        if res.x is not None:
            sc = _score(res.x)
            if sc < best_s:
                best_s = sc
                best_x = res.x
        if time.time() > deadline:
            break
    if best_x is None:
        best_x = starts[0]
        best_s = _score(best_x)
    return best_x
# EVOLVE-BLOCK-END
