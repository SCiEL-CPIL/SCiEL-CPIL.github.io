"""QAOA parameter design: per-graph numerical optimization with exact simulation."""
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = time.time() + 120.0
_EPS = 1e-9


def _time_left():
    return _DEADLINE - time.time()


def _build_cost_diag(edges, n):
    diag = np.zeros(n)
    for a, b in edges:
        diag[a] += 1.0
        diag[b] += 1.0
    return diag


def _expected_cut_value(params, cost_diag, n):
    """Exact expected cut value for a 2-layer QAOA circuit."""
    g1, b1, g2, b2 = params
    state = np.full(1 << n, 1.0 / np.sqrt(2.0 ** n))
    # layer 1 cost
    state = state * np.exp(-1j * g1 * cost_diag)
    # layer 1 mixer
    c1, s1 = np.cos(b1), np.sin(b1)
    state = c1 * state + 1j * s1 * np.conj(state)
    # layer 2 cost
    state = state * np.exp(-1j * g2 * cost_diag)
    # layer 2 mixer
    c2, s2 = np.cos(b2), np.sin(b2)
    state = c2 * state + 1j * s2 * np.conj(state)
    probs = np.abs(state) ** 2
    return float(np.dot(probs, cost_diag))


def _neg_value(params, cost_diag, n):
    return -_expected_cut_value(params, cost_diag, n)


def angles(n, edges, p):
    p = int(p)
    if _time_left() < 3.0:
        return np.tile([0.35, 0.25], (p, 1))

    cost_diag = _build_cost_diag(edges, n)
    dim = 1 << n

    # Coarse grid to find a good starting point.
    grid = np.linspace(0.0, np.pi, 17)
    best_val = -np.inf
    best_params = np.array([0.35, 0.25, 0.35, 0.25])
    for g1 in grid:
        for b1 in grid:
            for g2 in grid:
                for b2 in grid:
                    if _time_left() < 10.0:
                        break
                    val = _expected_cut_value([g1, b1, g2, b2], cost_diag, n)
                    if val > best_val:
                        best_val = val
                        best_params = np.array([g1, b1, g2, b2])
                if _time_left() < 10.0:
                    break
            if _time_left() < 10.0:
                break
        if _time_left() < 10.0:
            break

    # Local refinement with Nelder-Mead.
    if _time_left() > 15.0:
        try:
            res = minimize(_neg_value, best_params.copy(), args=(cost_diag, n),
                           method='Nelder-Mead',
                           options={'maxiter': 600, 'xatol': 1e-6, 'fatol': 1e-7})
            if res.success or res.fun < _neg_value(best_params, cost_diag, n):
                best_params = res.x
        except Exception:
            pass

    best_params = np.asarray(best_params, dtype=float)
    best_params = np.clip(best_params, 0.0, np.pi)

    if p == 2:
        return best_params.reshape(2, 2)
    # General p: tile the 2-layer schedule.
    if p <= 2:
        return best_params[:2 * p].reshape(p, 2)
    return np.tile(best_params.reshape(2, 2), ((p + 1) // 2, 1))[:p]
