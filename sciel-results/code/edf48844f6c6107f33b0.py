"""Suite task: pack 100 equal spheres in the unit cube [-1/2,1/2]^3.
Entry point construct() -> (N,3) array of centres.
Score = min( min_{i<j} |c_i-c_j|/2 , min_i min_k (1/2 - |c_ik|) ).
Design: repulsion-energy local optimization (L-BFGS-B, analytical grads)
from multiple diverse initializations + a final high-precision polish,
all under an explicit wall-clock deadline guard."""
import time
import numpy as np
from scipy.optimize import minimize

N = 100
_DEADLINE = 140.0          # safety margin vs 150 s wall-clock kill
_rng = np.random.default_rng(12345)


def _radius(c):
    """Common radius: min of half min pairwise distance and min wall clearance."""
    c = np.asarray(c, dtype=float)
    n = c.shape[0]
    d = 1e30
    for i in range(n - 1):
        diff = c[i + 1:] - c[i]
        d = min(d, float(np.min(np.einsum('ij,ij->i', diff, diff))))
    if d > 1e29:
        return float(np.min(0.5 - np.abs(c)))
    return min(0.5 * np.sqrt(d), float(np.min(0.5 - np.abs(c))))


def _energy(x, c, K, beta):
    """E = -K*min_pair_dist + beta*sum(wall_pen^2).  Grads analytic."""
    n = x.size
    c = np.asarray(c, dtype=float)
    # --- wall penalty (with gradient) ---
    pen = np.maximum(c - x, 0.0) + np.maximum(-c - x, 0.0)   # per coord
    e_wall = beta * float(np.dot(pen, pen))
    g_wall = -2.0 * beta * (np.sign(x - c) * np.maximum(c - x, 0.0)
                            + np.sign(-c - x) * np.maximum(-c - x, 0.0))
    # --- repulsion (soft-min of pairwise distances) ---
    d = np.zeros(n)
    g = np.zeros(n)
    for i in range(n - 1):
        diff = x[i + 1:] - x[i]
        dist = np.sqrt(np.einsum('ij,ij->i', diff, diff))
        d[i + 1:] = dist
        g[i + 1:] = diff / dist
    dmin = float(np.min(d))
    m = -np.logaddexp.reduce(-beta * d)
    e_rep = -K * m
    w = np.exp(-beta * d - m)
    g = g * w[:, None]
    e_rep += K * float(np.sum(w))
    g_rep = K * g
    return e_rep + e_wall, g_rep + g_wall


def _init_grid():
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    g = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    return g[:N].copy()


def _init_random():
    return _rng.uniform(-0.49, 0.49, size=(N, 3))


def _init_jittered():
    return _init_grid() + _rng.normal(0, 0.012, size=(N, 3))


def construct():
    t0 = time.time()
    best = _init_grid()
    best_r = _radius(best)

    inits = [_init_grid(), _init_random(), _init_jittered(), _init_random()]
    K, beta = 1.0, 80.0

    for it, c in enumerate(inits):
        if time.time() - t0 > _DEADLINE:
            break
        c = np.clip(c, -0.49, 0.49)
        x0 = c.ravel()
        try:
            res = minimize(_energy, x0, args=(c, K, beta), method='L-BFGS-B',
                           jac=True, options={'maxiter': 400, 'maxls': 30,
                                              'gtol': 1e-8, 'ftol': 1e-12})
            cand = np.clip(res.x.reshape(N, 3), -0.5, 0.5)
            r = _radius(cand)
            if r > best_r + 1e-12:
                best, best_r = cand, r
        except Exception:
            pass

    # --- final high-precision polish from the best point ---
    if time.time() - t0 < _DEADLINE - 20.0:
        c = best.copy()
        for _ in range(2):
            if time.time() - t0 > _DEADLINE:
                break
            try:
                res = minimize(_energy, c.ravel(), args=(c, K, beta),
                               method='L-BFGS-B', jac=True,
                               options={'maxiter': 3000, 'maxls': 50,
                                        'gtol': 1e-12, 'ftol': 1e-15})
                cand = np.clip(res.x.reshape(N, 3), -0.5, 0.5)
                r = _radius(cand)
                if r > best_r + 1e-14:
                    best, best_r = cand, r
            except Exception:
                break

    # --- final validity guard: clamp strictly inside, keep finite ---
    best = np.clip(best, -0.4999999, 0.4999999)
    if not np.all(np.isfinite(best)):
        best = _init_grid()
    return best
