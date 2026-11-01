"""Minimize logarithmic energy of 100 points on the unit sphere."""
import time
import math
import numpy as np

N = 100
DEADLINE = 150.0          # wall-clock limit (s); evaluator kills at 150s
SAFETY = 8.0              # return this many seconds before the limit
START_TIME = time.time()

# A tiny floor so log(1/d) stays finite if two points ever coincide.
# 1e-12 keeps the objective essentially exact for any reasonable config.
D_FLOOR = 1e-12


def _elapsed():
    return time.time() - START_TIME


def _remaining():
    return DEADLINE - SAFETY - _elapsed()


def _energy_and_grad(X):
    """Return (energy, grad) where grad has shape (N,3).

    Energy = sum_{i<j} log(1 / |x_i - x_j|)  (chordal, natural log).
    grad_i = - sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    (this is the gradient in R^3; caller projects to the tangent plane).
    """
    # Pairwise differences: D[i,j] = x_i - x_j
    D = X[:, None, :] - X[None, :, :]          # (N,N,3)
    dist2 = np.einsum('ijk,ijk->ij', D, D)     # (N,N)
    np.fill_diagonal(dist2, 1.0)               # avoid div-by-zero on diagonal
    dist = np.sqrt(dist2)
    # mask the diagonal
    mask = ~np.eye(N, dtype=bool)
    dist = np.where(mask, dist, 1.0)
    dist_safe = np.maximum(dist, D_FLOOR)

    # Energy: sum over i<j of log(1/dist) = -0.5 * sum_{i!=j} log(dist)
    logd = np.log(dist_safe)
    logd = np.where(mask, logd, 0.0)
    energy = -0.5 * logd.sum()

    # grad_i = - sum_{j!=i} (x_i - x_j) / dist^2
    # weight w[i,j] = 1/dist^2 for j != i
    w = np.where(mask, 1.0 / np.maximum(dist2, D_FLOOR * D_FLOOR), 0.0)
    grad = -np.einsum('ij,ijk->ik', w, D)      # (N,3)

    return energy, grad


def _project_tangent(X, grad):
    """Project grad onto tangent plane of the sphere at each x_i."""
    norm2 = np.einsum('ij,ij->i', X, X)
    norm2 = np.maximum(norm2, 1e-30)
    # tangent projection: g - (g.x) x / |x|^2
    gx = np.einsum('ij,ij->i', grad, X)
    return grad - (gx / norm2)[:, None] * X


def _reproject(X):
    """Normalize rows of X to unit length."""
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n = np.maximum(n, 1e-30)
    return X / n


def _objective_flat(flat):
    """Objective + gradient for L-BFGS-B on flattened (N*3) vector.

    We minimize the energy but keep points on the sphere by re-projecting
    inside the objective and projecting the gradient to the tangent plane.
    L-BFGS-B operates in R^{3N}; the re-projection makes the mapping
    idempotent so the effective search stays on the sphere manifold.
    """
    X = _reproject(flat.reshape(N, 3))
    e, g = _energy_and_grad(X)
    gt = _project_tangent(X, g)
    return e, gt.ravel()


def _fibonacci_start():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_start(rng):
    v = rng.standard_normal((N, 3))
    return _reproject(v)


def _optimize_from(X0, max_iter=4000, tol=1e-12, time_budget=None):
    """Run L-BFGS-B from X0; return (energy, X) of the best point found."""
    from scipy.optimize import minimize

    flat0 = X0.ravel()
    best_e, best_flat = _objective_flat(flat0)
    best_X = _reproject(flat0.reshape(N, 3))

    # adaptive time budget per optimization
    if time_budget is None:
        time_budget = max(2.0, min(20.0, _remaining() * 0.5))

    t0 = time.time()

    def _wrapped(flat):
        e, g = _objective_flat(flat)
        if e < best_e:
            pass  # updated below
        return e, g

    # We need to track best; use a closure.
    state = {'e': best_e, 'flat': best_flat.copy()}

    def _obj(flat):
        e, g = _objective_flat(flat)
        if e < state['e']:
            state['e'] = e
            state['flat'] = flat.copy()
        return e, g

    res = minimize(
        _obj, flat0,
        method='L-BFGS-B',
        options={'maxiter': max_iter, 'gtol': tol, 'ftol': 1e-15,
                 'maxfun': 200000},
    )
    # evaluate the returned point too
    e_r, _ = _objective_flat(res.x)
    if e_r < state['e']:
        state['e'] = e_r
        state['flat'] = res.x.copy()

    best_X = _reproject(state['flat'].reshape(N, 3))
    return state['e'], best_X


def construct():
    """Return array of shape (N,3) of unit vectors minimizing log energy."""
    rng = np.random.default_rng(12345)

    starts = []
    # Fibonacci spiral as a solid baseline start
    starts.append(_fibonacci_start())
    # Several random starts
    n_random = 12
    for _ in range(n_random):
        starts.append(_random_start(rng))

    best_e = np.inf
    best_X = starts[0]

    for k, X0 in enumerate(starts):
        if _remaining() < 3.0:
            break
        # give later starts a bit less budget
        per = max(1.0, min(15.0, _remaining() * 0.4))
        e, X = _optimize_from(X0, max_iter=3000, tol=1e-12, time_budget=per)
        if e < best_e:
            best_e = e
            best_X = X

    # Final high-precision polish from the best configuration
    if _remaining() > 4.0:
        e, X = _optimize_from(best_X, max_iter=8000, tol=1e-14,
                              time_budget=max(2.0, _remaining() * 0.7))
        if e < best_e:
            best_e = e
            best_X = X

    # Final Newton-style polish: a few explicit gradient steps with
    # adaptive line search on the tangent plane, very tight.
    if _remaining() > 2.0:
        X = _reproject(best_X)
        for _ in range(60):
            if _remaining() < 1.0:
                break
            e, g = _energy_and_grad(X)
            gt = _project_tangent(X, g)
            gn = np.linalg.norm(gt)
            if gn < 1e-14:
                break
            # backtracking line search
            step = 1.0
            improved = False
            for _ls in range(30):
                Xn = _reproject(X - step * gt)
                en, _ = _energy_and_grad(Xn)
                if en < e:
                    X = Xn
                    e = en
                    improved = True
                    break
                step *= 0.5
            if not improved:
                break

    X = _reproject(best_X)

    # Safety: ensure finite and no near-coincident points
    if not np.all(np.isfinite(X)):
        X = _fibonacci_start()
    # check pairwise distances
    D = X[:, None, :] - X[None, :, :]
    d = np.linalg.norm(D, axis=2)
    np.fill_diagonal(d, np.inf)
    if d.min() < 1e-9:
        # nudge: fall back to a fresh random+optimize
        X = _optimize_from(_random_start(rng), max_iter=2000, tol=1e-12)[1]
        X = _reproject(X)

    return X
