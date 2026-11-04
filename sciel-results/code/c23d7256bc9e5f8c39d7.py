"""New design: Riesz s-energy + boundary repulsion minimized with L-BFGS-B from FCC init."""
N = 100
import time
import numpy as np
from scipy.optimize import minimize

_START = None
_DEADLINE = None


def _elapsed():
    global _START
    if _START is None:
        _START = time.time()
    return time.time() - _START


def _remaining():
    global _DEADLINE
    if _DEADLINE is None:
        _DEADLINE = 125.0  # safety margin before 150s kill
    return _DEADLINE - _elapsed()


def _fcc_points(n, cube=0.5):
    """Generate n points on a face-centered cubic lattice, centered at origin, within [-cube, cube]."""
    # FCC lattice vectors
    a = 1.0
    basis = np.array([
        [0, 0, 0],
        [0, a / 2, a / 2],
        [a / 2, 0, a / 2],
        [a / 2, a / 2, 0],
    ])
    pts = []
    R = int(np.ceil(np.cbrt(n))) + 2
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            for k in range(-R, R + 1):
                for b in basis:
                    p = np.array([i, j, k]) * a + b
                    pts.append(p)
    pts = np.array(pts)
    # Center the cloud at origin
    pts -= pts.mean(axis=0)
    # Scale so the cloud fits inside [-cube, cube] with margin
    extent = np.max(np.abs(pts), axis=0)
    scale = (cube * 0.98) / np.max(extent) if np.max(extent) > 0 else 1.0
    pts *= scale
    # Shuffle deterministically and take n
    rng = np.random.RandomState(12345)
    idx = rng.permutation(len(pts))[:n]
    return pts[idx]


def _energy_and_grad(pts, s=8.0, beta=30.0, cube=0.5, eps=1e-4):
    """Riesz s-energy (pairwise repulsion) + boundary repulsion. Returns (energy, grad)."""
    n = pts.shape[0]
    pts = np.asarray(pts, dtype=float)
    # Pairwise distances
    diff = pts[:, None, :] - pts[None, :, :]  # (n, n, 3)
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(d2)
    # Riesz s-energy: sum_{i<j} 1/d^s  (repulsive, s>3 favors uniform spacing)
    # E_pair = 0.5 * sum_{i != j} d^{-s}
    inv_ds = d2 ** (-(s / 2.0))
    np.fill_diagonal(inv_ds, 0.0)
    E_pair = 0.5 * np.sum(inv_ds)
    # Gradient: d/dx_i [0.5 * sum_j (x_i - x_j)^-s] = 0.5 * sum_j (-s) (x_i - x_j) |x_i-x_j|^{-s-2}
    # = -0.5 * s * sum_j (x_i - x_j) * d2^{-(s/2 + 1)}
    d2_safe = d2.copy()
    np.fill_diagonal(d2_safe, 1.0)
    w = d2_safe ** (-(s / 2.0 + 1.0))
    np.fill_diagonal(w, 0.0)
    # grad_pair[i] = -0.5 * s * sum_j (pts[i] - pts[j]) * w[i,j]
    # = -0.5 * s * ( pts[i] * sum_j w[i,j] - sum_j pts[j] * w[i,j] )
    row_sum = np.sum(w, axis=1)  # (n,)
    grad_pair = -0.5 * s * (pts * row_sum[:, None] - (w @ pts))
    # Boundary repulsion: penalize |x_k| approaching cube. Use smooth soft penalty.
    # For each coordinate, penalty = (cube - |x|) when x is inside, else large.
    # Use: for each axis, q = cube - |x| (distance to wall, positive inside).
    # Penalty term per axis: 1/q^beta for q>0 (repulsive wall), smooth.
    # We'll compute per-coordinate.
    ax = np.abs(pts)  # (n,3)
    q = cube - ax  # (n,3) distance to nearest wall in that axis
    # Avoid division by zero: clip q to eps
    q_safe = np.clip(q, eps, None)
    E_wall = np.sum(q_safe ** (-beta))
    # grad: d/dx_{i,k} [q^{-beta}] where q = cube - |x|
    # d q/d x = -sign(x) (when x != 0); d/dx [q^{-beta}] = -beta q^{-beta-1} * (-sign(x)) = beta q^{-beta-1} sign(x)
    sign = np.sign(pts)
    grad_wall = beta * q_safe ** (-(beta + 1.0)) * sign
    total = E_pair + E_wall
    grad = grad_pair + grad_wall
    return total, grad


def construct():
    global _START, _DEADLINE
    _START = time.time()
    _DEADLINE = 125.0

    # FCC-based dense initialization
    pts = _fcc_points(N)
    best = pts.copy()
    best_score = _score(best)

    s = 8.0
    beta = 30.0
    cube = 0.5

    # L-BFGS-B with analytic gradient
    def fun(x):
        p = x.reshape(N, 3)
        e, _ = _energy_and_grad(p, s, beta, cube)
        return e

    def jac(x):
        p = x.reshape(N, 3)
        _, g = _energy_and_grad(p, s, beta, cube)
        return g.ravel()

    x0 = pts.ravel()
    bounds = [(-cube, cube)] * (3 * N)

    try:
        res = minimize(fun, x0, jac=jac, method='L-BFGS-B',
                       bounds=bounds, options={'maxiter': 2000, 'gtol': 1e-8,
                                               'maxls': 50, 'disp': False})
        if res.x is not None and np.all(np.isfinite(res.x)):
            cand = res.x.reshape(N, 3)
            sc = _score(cand)
            if sc > best_score:
                best = cand
                best_score = sc
    except Exception:
        pass

    if _remaining() < 5.0:
        return _clamp(best)

    # Second phase: lower s and beta to refine, starting from best
    for (s2, beta2) in [(6.0, 20.0), (5.0, 15.0)]:
        if _remaining() < 8.0:
            break
        try:
            x0 = best.ravel()

            def fun2(x, s=s2, b=beta2):
                p = x.reshape(N, 3)
                e, _ = _energy_and_grad(p, s, b, cube)
                return e

            def jac2(x, s=s2, b=beta2):
                p = x.reshape(N, 3)
                _, g = _energy_and_grad(p, s, b, cube)
                return g.ravel()

            res = minimize(fun2, x0, jac=jac2, method='L-BFGS-B',
                           bounds=bounds, options={'maxiter': 1500, 'gtol': 1e-9,
                                                   'maxls': 50, 'disp': False})
            if res.x is not None and np.all(np.isfinite(res.x)):
                cand = res.x.reshape(N, 3)
                sc = _score(cand)
                if sc > best_score:
                    best = cand
                    best_score = sc
        except Exception:
            pass

    return _clamp(best)


def _clamp(pts):
    """Clamp centers strictly inside the cube to guarantee validity."""
    pts = np.asarray(pts, dtype=float)
    m = 1e-9
    pts = np.clip(pts, -0.5 + m, 0.5 - m)
    return pts


def _score(pts):
    """Compute the feasible radius r = min(min_{i<j}|ci-cj|/2, min_i min_k (1/2 - |c_ik|))."""
    pts = np.asarray(pts, dtype=float)
    n = pts.shape[0]
    # Wall distances
    wall = 0.5 - np.max(np.abs(pts), axis=1)
    r_wall = np.min(wall)
    # Pairwise
    diff = pts[:, None, :] - pts[None, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    dmin = np.sqrt(np.min(d2))
    r_pair = dmin / 2.0
    return float(min(r_wall, r_pair))
