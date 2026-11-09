"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import math
import time

N = 150


def _fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _pairwise_dists(X):
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    return D


def _energy(X):
    D = _pairwise_dists(X)
    return math.fsum(math.log(1.0 / d) for d in D[D < np.inf].ravel())


def _gradient(X):
    D = _pairwise_dists(X)
    inv = 1.0 / D
    inv[np.diag_indices_from(inv)] = 0.0
    # grad_i = sum_{j != i} x_i / |x_i - x_j|
    return X * inv.sum(axis=1)[:, None]


def _normalize(X):
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    return X / nrm


def construct():
    t0 = time.time()
    deadline = t0 + 140.0  # safety margin before 150 s kill

    X = _fibonacci_points(N).astype(np.float64)
    best_X = X.copy()
    best_E = _energy(X)

    rng = np.random.default_rng(12345)

    # Phase 1: gradient descent with momentum + annealing kicks.
    step = 5e-4
    momentum = 0.0
    v = np.zeros_like(X)
    kick = 0.0
    kick_decay = 0.985
    kick_min = 0.0
    step_decay = 0.9995
    step_min = 1e-7
    it = 0
    while time.time() < deadline - 15.0:  # reserve ~15 s for final polish
        G = _gradient(X)
        v = momentum * v - step * G
        X = _normalize(X + v)
        E = _energy(X)
        if E < best_E:
            best_E = E
            best_X = X.copy()
        # simulated-annealing random kick to escape local minima
        if kick > kick_min:
            X = _normalize(X + kick * rng.standard_normal(X.shape))
        step = max(step_min, step * step_decay)
        kick = max(kick_min, kick * kick_decay)
        it += 1
        if it % 200 == 0:
            # re-inject a fresh kick to keep exploring
            kick = max(kick, 1e-3)

    # Phase 2: final L-BFGS-B polish from the best point found.
    X = best_X.copy()

    def f_and_g(flat):
        Xl = _normalize(flat.reshape(N, 3))
        E = _energy(Xl)
        G = _gradient(Xl)
        return E, G.ravel()

    try:
        from scipy.optimize import minimize
        state = [None]

        def obj(flat):
            E, g = f_and_g(flat)
            state[0] = g
            return E

        def gobj(flat):
            if state[0] is None:
                return f_and_g(flat)[1]
            return state[0]

        res = minimize(
            obj,
            X.ravel(),
            jac=gobj,
            method="L-BFGS-B",
            options={"maxiter": 200, "gtol": 1e-9, "maxls": 40},
        )
        X2 = _normalize(res.x.reshape(N, 3))
        E2 = _energy(X2)
        if E2 < best_E:
            best_E = E2
            best_X = X2
    except Exception:
        pass

    # Safety: ensure valid output (finite, unit length, distinct points).
    out = _normalize(best_X)
    if (not np.all(np.isfinite(out))) or np.any(np.linalg.norm(out, axis=1) == 0):
        out = _normalize(_fibonacci_points(N))
    return out
