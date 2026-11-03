"""Rewritten program: minimize logarithmic energy of 100 points on the unit sphere.

Design (discrete points, smooth continuous optimization, high-precision polish):
  * Start from a Fibonacci (golden-angle) spiral, a good deterministic spread.
  * Optimize directly on the sphere using an analytic gradient, with the sphere
    constraint handled by projection (a change-of-variables that removes the
    constraint from the search).
  * Alternating robust optimizers (L-BFGS-B with backtracking, and momentum
    gradient descent) with multiple restarts to escape shallow local minima.
  * A final high-precision polish pass with very tight tolerances.
  * An explicit wall-clock deadline guard: always return the best configuration
    found with a safety margin before the 150 s limit.
"""

import time
import math
import numpy as np

N = 100

# ---------------------------------------------------------------------------
# Deadline guard: the evaluator kills at 150 s wall-clock.  Keep a safety margin.
# ---------------------------------------------------------------------------
_START = time.time()
_DEADLINE = 150.0
_SAFETY = 8.0          # seconds to leave before the hard limit
_DEADLINE_HARD = _DEADLINE - _SAFETY


def _time_left():
    return _DEADLINE_HARD - (time.time() - _START)


# ---------------------------------------------------------------------------
# Energy and analytic gradient (chordal distances, natural log).
#   E = sum_{i<j} log(1 / |x_i - x_j|) = - sum_{i<j} log |x_i - x_j|
#   grad_i E = - sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
# ---------------------------------------------------------------------------
def _energy_and_grad(X):
    """X: (N,3) array of points (need not be exactly unit, but should be close).
    Returns (energy, grad) where grad is the Euclidean gradient of E w.r.t. X."""
    # Pairwise difference matrix D[i,j] = x_i - x_j, shape (N,N,3)
    D = X[:, None, :] - X[None, :, :]          # (N, N, 3)
    dist = np.linalg.norm(D, axis=2)           # (N, N)
    np.fill_diagonal(dist, 1.0)                # avoid divide by zero on diagonal

    # log(1/d) = -log d  for i != j ; diagonal contributes 0
    logd = np.log(dist)
    np.fill_diagonal(logd, 0.0)
    energy = -0.5 * float(np.sum(logd))        # sum over i<j = half of full sum

    # grad_i = - sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # 1/dist^2 with diagonal set to 0
    inv_d2 = 1.0 / (dist * dist)
    np.fill_diagonal(inv_d2, 0.0)
    # grad[i] = - sum_j inv_d2[i,j] * D[i,j]
    grad = -np.sum(inv_d2[:, :, None] * D, axis=1)   # (N, 3)
    return energy, grad


def _energy_only(X):
    return _energy_and_grad(X)[0]


def _project_to_sphere(X):
    """Normalize rows to unit length (guard against zero rows)."""
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm < 1e-12] = 1.0
    return X / nrm


def _tangent_grad(X, grad):
    """Project Euclidean gradient onto the tangent space of the sphere at each point."""
    X = _project_to_sphere(X)
    return grad - (grad * X).sum(axis=1, keepdims=True) * X


# ---------------------------------------------------------------------------
# Initial configuration: Fibonacci (golden-angle) spiral.
# ---------------------------------------------------------------------------
def _golden_spiral():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0.0, 1.0))
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _project_to_sphere(X)


# ---------------------------------------------------------------------------
# Optimizer 1: L-BFGS-B with a backtracking (Armijo) line search, on the sphere.
# We optimize the unconstrained Euclidean variables but project to the sphere
# after each step and use the tangent gradient (constraint removed by projection).
# ---------------------------------------------------------------------------
def _lbfgs_sphere(X0, max_iter, deadline_check):
    from scipy.optimize import minimize
    X = _project_to_sphere(X0.copy())
    best = X.copy()
    best_e = _energy_only(X)

    def fun_and_grad(vec):
        Xc = _project_to_sphere(vec.reshape(N, 3))
        e, g = _energy_and_grad(Xc)
        gt = _tangent_grad(Xc, g)
        return e, gt.reshape(-1)

    # Wrap with projection: L-BFGS-B on the raw vector, we project inside.
    x0 = X.reshape(-1)
    res = minimize(
        fun_and_grad,
        x0,
        method="L-BFGS-B",
        jac=True,
        options={
            "maxiter": max_iter,
            "ftol": 1e-15,
            "gtol": 1e-12,
            "maxls": 50,
            "maxfun": 4000,
        },
        callback=lambda v: (deadline_check() < 0),
    )
    Xc = _project_to_sphere(res.x.reshape(N, 3))
    e = _energy_only(Xc)
    if e < best_e:
        best, best_e = Xc, e
    return best, best_e


# ---------------------------------------------------------------------------
# Optimizer 2: momentum gradient descent with backtracking Armijo line search,
# operating in tangent space (constraint removed by reparametrization).
# ---------------------------------------------------------------------------
def _momentum_gd(X0, max_iter, deadline_check):
    X = _project_to_sphere(X0.copy())
    best = X.copy()
    best_e = _energy_only(X)
    v = np.zeros_like(X)
    lr = 0.02
    momentum = 0.9
    c_armijo = 1e-4
    rho = 0.5
    eps = 1e-12

    for it in range(max_iter):
        if deadline_check() < 0:
            break
        e, g = _energy_and_grad(X)
        gt = _tangent_grad(X, g)
        v = momentum * v + lr * gt
        # backtracking line search in tangent direction
        step = v
        for _ in range(40):
            Xn = _project_to_sphere(X + step)
            en = _energy_only(Xn)
            if en <= e + c_armijo * np.sum(gt * step):
                break
            step *= rho
        else:
            break
        X = Xn
        e = en
        if e < best_e:
            best, best_e = Xn.copy(), e
        # adaptive learning rate: shrink if we barely moved
        if it % 200 == 199 and np.linalg.norm(v) < 1e-9:
            lr *= 0.5
    return best, best_e


# ---------------------------------------------------------------------------
# Optimizer 3: final high-precision polish with very small fixed steps.
# ---------------------------------------------------------------------------
def _polish(X0, max_iter, deadline_check):
    X = _project_to_sphere(X0.copy())
    best = X.copy()
    best_e = _energy_only(X)
    lr = 1e-4
    for it in range(max_iter):
        if deadline_check() < 0:
            break
        e, g = _energy_and_grad(X)
        gt = _tangent_grad(X, g)
        gnorm = np.linalg.norm(gt)
        if gnorm < 1e-14:
            break
        step = -lr * gt / (gnorm + 1e-300) * (1e-4)   # tiny normalized step
        Xn = _project_to_sphere(X + step)
        en = _energy_only(Xn)
        if en < best_e:
            best, best_e = Xn.copy(), en
        X = Xn
    return best, best_e


# ---------------------------------------------------------------------------
# Main entry point.
# ---------------------------------------------------------------------------
def construct():
    deadline_check = lambda: _time_left()

    # Deterministic base configuration.
    X0 = _golden_spiral()
    best = X0.copy()
    best_e = _energy_only(X0)

    rng = np.random.default_rng(12345)

    def maybe_update(X, e):
        nonlocal best, best_e
        if e < best_e:
            best, best_e = _project_to_sphere(X.copy()), e

    # --- Phase 1: strong L-BFGS-B from the spiral -------------------------
    if deadline_check() > 10:
        X1, e1 = _lbfgs_sphere(X0, max_iter=600, deadline_check=deadline_check)
        maybe_update(X1, e1)

    # --- Phase 2: momentum GD from the best so far ------------------------
    if deadline_check() > 10:
        X2, e2 = _momentum_gd(best, max_iter=4000, deadline_check=deadline_check)
        maybe_update(X2, e2)

    # --- Phase 3: a few random-perturbation restarts to escape local minima -
    # Each restart: perturb the current best, run a short momentum-GD, keep best.
    perturb_scale = 0.05
    for k in range(8):
        if deadline_check() < 25:
            break
        noise = rng.normal(0.0, perturb_scale, size=best.shape)
        Xp = _project_to_sphere(best + noise)
        Xr, er = _momentum_gd(Xp, max_iter=2500, deadline_check=deadline_check)
        maybe_update(Xr, er)
        # anneal the perturbation down so later restarts are finer
        perturb_scale *= 0.6

    # --- Phase 4: final high-precision polish -----------------------------
    if deadline_check() > 15:
        Xf, ef = _polish(best, max_iter=20000, deadline_check=deadline_check)
        maybe_update(Xf, ef)

    # --- Final validation / safety: ensure unit vectors, finite, distinct --
    Xout = _project_to_sphere(best)
    Xout = np.nan_to_num(Xout, nan=0.0, posinf=0.0, neginf=0.0)
    # Re-normalize once more to be safe.
    Xout = _project_to_sphere(Xout)
    # Ensure no zero rows (should not happen after projection, but guard).
    for i in range(N):
        if np.linalg.norm(Xout[i]) < 1e-12:
            Xout[i] = np.array([1.0, 0.0, 0.0])
    return Xout
